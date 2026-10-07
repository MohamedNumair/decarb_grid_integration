# DECARB Study Runner — Sequential Cases, Parallel Buildings
# Spawns workers once, reuses across all cases.
# Uses @spawnat + RemoteChannel (not pmap) for reliable dispatch.

using Distributed, CSV, DataFrames, Dates

# ── Settings ──────────────────────────────────────────────────────────────
STUDY_DIR        = raw"D:\shared\building_grid_integration\out\studies\study_1\decarb"
DECARB_SRC       = raw"C:\decarb\src\DECARB.jl"
LOG_FILE         = joinpath(STUDY_DIR, "run_decarb.log")
UPDATE_FOLDER    = "update_0"
N_WORKERS        = 20
GUROBI_THREADS   = 1
I_SOLVER         = 1
RELAX            = false
SKIP_COMPLETED   = true
OVERWRITE_FAILED = false
STALL_TIMEOUT    = 900        # seconds with no result before skipping case
# ──────────────────────────────────────────────────────────────────────────

function fmt(seconds)
    h, r = divrem(floor(Int, seconds), 3600)
    m, s = divrem(r, 60)
    return "$(h)h $(m)m $(s)s"
end

function log_msg(msg)
    ts = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    line = "[$ts] $msg"
    println(line)
    flush(stdout)
    open(LOG_FILE, "a") do io
        println(io, line)
        flush(io)
    end
end

# ── Kill any existing workers, then spin up fresh ones ────────────────────
rmprocs(workers())
addprocs(N_WORKERS; exeflags = "--project=$(Base.active_project())")
log_msg("Launched $(nworkers()) worker(s).")

# ── Load DECARB on every worker (once, before the case loop) ─────────────
log_msg("Loading DECARB on workers (please wait)...")
@everywhere begin
    ENV["GRB_THREADS"] = string($GUROBI_THREADS)
    using CSV, DataFrames, Dates
    include($(DECARB_SRC))
end
log_msg("Workers ready.")

# ── Per-building function (defined once on all workers) ───────────────────
@everywhere function run_one(bdg, case_dir, update_folder, i_solver, relax,
                              gurobi_threads)
    t    = time()
    path = joinpath(case_dir, bdg, update_folder)
    try
        mkpath(joinpath(path, "out"))
        ENV["GRB_THREADS"] = string(gurobi_threads)
        redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                DECARB.run_decarb!(path, i_solver, relax)
            end
        end
        elapsed_s = round(time() - t; digits = 2)
        return (building = bdg, status = :ok, elapsed_s = elapsed_s, err = "")
    catch e
        return (building = bdg, status = :error,
                elapsed_s = round(time() - t; digits = 2),
                err = sprint(showerror, e))
    end
end

@everywhere function run_one_ch(bdg, case_dir, update_folder, i_solver, relax,
                                 gurobi_threads, ch)
    result = run_one(bdg, case_dir, update_folder, i_solver, relax,
                     gurobi_threads)
    put!(ch, result)
    return result
end

# ── Discover all cases ────────────────────────────────────────────────────
all_cases = sort(filter(d -> isdir(joinpath(STUDY_DIR, d)), readdir(STUDY_DIR)))
log_msg("Found $(length(all_cases)) cases: $(join(all_cases, ", "))")

# ── Run each case sequentially ────────────────────────────────────────────
for (case_idx, case_name) in enumerate(all_cases)
    case_dir   = joinpath(STUDY_DIR, case_name)
    failed_csv = joinpath(case_dir, "failed_buildings.csv")

    log_msg("═"^70)
    log_msg("CASE $case_idx / $(length(all_cases)): $case_name")
    log_msg("═"^70)

    # ── Build skip set from previous failures ─────────────────────────
    skip_ids = Set{String}()

    if OVERWRITE_FAILED && isfile(failed_csv)
        rm(failed_csv)
        log_msg("  Cleared failed_buildings.csv")
    elseif isfile(failed_csv)
        prev = CSV.read(failed_csv, DataFrame; header = 1, normalizenames = true)
        union!(skip_ids, Set(string.(prev.bldg_id)))
        log_msg("  Loaded $(length(skip_ids)) previously failed buildings")
    end

    # ── Discover buildings ────────────────────────────────────────────
    buildings = filter(
        d -> isdir(joinpath(case_dir, d, UPDATE_FOLDER, "in")) && d ∉ skip_ids,
        readdir(case_dir)
    )

    if SKIP_COMPLETED
        before = length(buildings)
        buildings = filter(
            d -> !isfile(joinpath(case_dir, d, UPDATE_FOLDER, "out", "ts.csv")),
            buildings
        )
        log_msg("  Skipping completed: $(before - length(buildings)) already done")
    end

    n_total = length(buildings)

    if n_total == 0
        log_msg("  Nothing to run — all buildings completed or skipped.")
        continue
    end

    log_msg("  Buildings to run: $n_total | Workers: $(nworkers())")

    # ── Dispatch via @spawnat + RemoteChannel ─────────────────────────
    ch      = RemoteChannel(() -> Channel{Any}(n_total))
    t_start = time()

    futures = [@spawnat :any run_one_ch(bdg, case_dir, UPDATE_FOLDER,
                                         I_SOLVER, RELAX, GUROBI_THREADS,
                                         ch)
               for bdg in buildings]

    log_path   = joinpath(case_dir, "batch_log_$(UPDATE_FOLDER).csv")
    n_received = 0

    # Collect results — stall timeout breaks out of the wait loop for
    # the current building, but keeps collecting any remaining results
    # that are still trickling in.  Only truly "stuck" buildings are
    # skipped, not the whole case.
    stall_count = 0
    while n_received < n_total
        r = nothing
        t_wait = time()
        while r === nothing
            if isready(ch)
                r = take!(ch)
            else
                sleep(5)
                if time() - t_wait > STALL_TIMEOUT
                    stall_count += 1
                    n_alive = nworkers()
                    log_msg("  ⚠  No results for $(STALL_TIMEOUT)s — $n_alive workers alive. (stall #$stall_count)")
                    log_msg("     Received $n_received / $n_total so far.")
                    break   # break the inner wait loop, not the outer collection loop
                end
            end
        end

        # If we broke out due to stall, check if anything else is ready
        # before giving up on this case entirely
        if r === nothing
            # Drain any results that came in while we were logging
            while isready(ch)
                r = take!(ch)
                n_received += 1
                elapsed     = time() - t_start
                eta         = n_received < n_total ? elapsed / n_received * (n_total - n_received) : 0.0
                icon        = r.status == :ok ? "✓" : "✗"
                log_msg("  $icon  $n_received / $n_total | $(fmt(elapsed)) | ETA: $(fmt(eta)) | $(r.building) | workers: $(nworkers())")

                log_row = DataFrame(building = [r.building], status = [string(r.status)],
                                    elapsed_s = [r.elapsed_s], err = [r.err])
                CSV.write(log_path, log_row; append = n_received > 1)

                if r.status == :error
                    row = DataFrame(bldg_id = [r.building], status = [string(r.status)],
                                    error = [r.err])
                    CSV.write(failed_csv, row; append = isfile(failed_csv))
                end
            end
            # If we've stalled MAX_STALLS times in a row with nothing,
            # the remaining buildings are truly hung — move on
            if stall_count >= 3
                n_stuck = n_total - n_received
                log_msg("  ⚠  $n_stuck building(s) appear stuck after $stall_count stalls — moving to next case.")
                break
            end
            continue
        end

        # Got a normal result — reset stall counter
        stall_count = 0
        n_received += 1
        elapsed     = time() - t_start
        eta         = n_received < n_total ? elapsed / n_received * (n_total - n_received) : 0.0
        icon        = r.status == :ok ? "✓" : "✗"
        log_msg("  $icon  $n_received / $n_total | $(fmt(elapsed)) | ETA: $(fmt(eta)) | $(r.building) | workers: $(nworkers())")

        # ── Write batch log immediately ───────────────────────────────
        log_row = DataFrame(building = [r.building], status = [string(r.status)],
                            elapsed_s = [r.elapsed_s], err = [r.err])
        if n_received == 1
            CSV.write(log_path, log_row)
        else
            CSV.write(log_path, log_row; append = true)
        end

        # ── Write failure immediately ─────────────────────────────────
        if r.status == :error
            row = DataFrame(bldg_id = [r.building], status = [string(r.status)],
                            error = [r.err])
            if isfile(failed_csv)
                CSV.write(failed_csv, row; append = true)
            else
                CSV.write(failed_csv, row)
            end
        end
    end

    # ── Case summary ──────────────────────────────────────────────────
    elapsed_total = time() - t_start
    log_msg("Case '$case_name' done in $(fmt(elapsed_total)) — $n_received / $n_total completed")

    # ── Deduplicate failed_buildings.csv ──────────────────────────────
    if isfile(failed_csv)
        fail_df = CSV.read(failed_csv, DataFrame; header = 1, normalizenames = true)
        unique!(fail_df, :bldg_id)
        CSV.write(failed_csv, fail_df)
    end
end

# ── Final summary ─────────────────────────────────────────────────────────
log_msg("═"^70)
log_msg("ALL CASES COMPLETE")
log_msg("═"^70)
for case_name in all_cases
    case_dir = joinpath(STUDY_DIR, case_name)
    done = count(
        f -> isfile(joinpath(case_dir, f, UPDATE_FOLDER, "out", "ts.csv")),
        readdir(case_dir)
    )
    log_msg("  $case_name: $done completed")
end
log_msg("Log file: $LOG_FILE")
