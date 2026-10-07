# DECARB Study Runner — Sequential Cases x Updates, Parallel Buildings
# Spawns workers once, reuses across all cases and all update_* folders.
# Uses @spawnat + RemoteChannel (not pmap) for reliable dispatch.

using Distributed, CSV, DataFrames, Dates

# ── Settings ──────────────────────────────────────────────────────────────
STUDY_DIR        = "/home/onurtalu/decarb"
DECARB_SRC       = joinpath(@__DIR__, "src", "DECARB.jl")
LOG_FILE         = joinpath(STUDY_DIR, "run_decarb.log")
N_WORKERS        = 60
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

# ── Discover every "update_N" subfolder present under a case, across all
#    of its building folders (buildings within a case can, in principle,
#    have slightly different update sets — we take the union) ────────────
function discover_update_folders(case_dir)
    updates = Set{String}()
    for d in readdir(case_dir)
        bldg_path = joinpath(case_dir, d)
        isdir(bldg_path) || continue
        for sub in readdir(bldg_path)
            if isdir(joinpath(bldg_path, sub)) && occursin(r"^update_\d+$", sub)
                push!(updates, sub)
            end
        end
    end
    return sort(collect(updates); by = s -> parse(Int, replace(s, "update_" => "")))
end

# ── Kill any existing workers, then spin up fresh ones ────────────────────
rmprocs(workers())
addprocs(N_WORKERS; exeflags = "--project=$(Base.active_project())")
log_msg("Launched $(nworkers()) worker(s).")

# ── Load DECARB on every worker (once, before the case loop) ─────────────
log_msg("Loading DECARB on workers (please wait)...")
@everywhere begin
    ENV["GRB_THREADS"] = string($GUROBI_THREADS)
    ENV["GRB_LICENSE_FILE"] = "/home/onurtalu/gurobi.lic"
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

# ── Run each case, and each update_* folder within it, sequentially ──────
for (case_idx, case_name) in enumerate(all_cases)
    case_dir = joinpath(STUDY_DIR, case_name)

    update_folders = discover_update_folders(case_dir)
    if isempty(update_folders)
        log_msg("CASE $case_idx / $(length(all_cases)): $case_name — no update_* folders found, skipping")
        continue
    end

    log_msg("═"^70)
    log_msg("CASE $case_idx / $(length(all_cases)): $case_name — $(length(update_folders)) update folder(s): $(join(update_folders, ", "))")
    log_msg("═"^70)

    for update_folder in update_folders
        failed_csv = joinpath(case_dir, "failed_buildings_$(update_folder).csv")

        log_msg("─"^70)
        log_msg("  UPDATE: $update_folder  (case: $case_name)")
        log_msg("─"^70)

        # ── Build skip set from previous failures (per case + update) ──
        skip_ids = Set{String}()

        if OVERWRITE_FAILED && isfile(failed_csv)
            rm(failed_csv)
            log_msg("  Cleared failed_buildings_$(update_folder).csv")
        elseif isfile(failed_csv)
            prev = CSV.read(failed_csv, DataFrame; header = 1, normalizenames = true)
            union!(skip_ids, Set(string.(prev.bldg_id)))
            log_msg("  Loaded $(length(skip_ids)) previously failed buildings")
        end

        # ── Discover buildings ────────────────────────────────────────
        buildings = filter(
            d -> isdir(joinpath(case_dir, d, update_folder, "in")) && d ∉ skip_ids,
            readdir(case_dir)
        )

        if SKIP_COMPLETED
            before = length(buildings)
            buildings = filter(
                d -> !isfile(joinpath(case_dir, d, update_folder, "out", "ts.csv")),
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

        # ── Dispatch via @spawnat + RemoteChannel ─────────────────────
        ch      = RemoteChannel(() -> Channel{Any}(n_total))
        t_start = time()

        futures = [@spawnat :any run_one_ch(bdg, case_dir, update_folder,
                                             I_SOLVER, RELAX, GUROBI_THREADS,
                                             ch)
                   for bdg in buildings]

        log_path   = joinpath(case_dir, "batch_log_$(update_folder).csv")
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

            # If we broke out due to stall, drain anything that arrived and decide
            if r === nothing
                while isready(ch)
                    r = take!(ch)
                    n_received += 1
                    stall_count = 0   # got a result, reset
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
                # 3 consecutive stalls with nothing → remaining buildings are hung
                if stall_count >= 3
                    n_stuck = n_total - n_received
                    log_msg("  ⚠  $n_stuck building(s) appear stuck after $stall_count stalls — moving to next update/case.")
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

            # ── Write batch log immediately ───────────────────────────
            log_row = DataFrame(building = [r.building], status = [string(r.status)],
                                elapsed_s = [r.elapsed_s], err = [r.err])
            if n_received == 1
                CSV.write(log_path, log_row)
            else
                CSV.write(log_path, log_row; append = true)
            end

            # ── Write failure immediately ─────────────────────────────
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

        # ── Update summary ──────────────────────────────────────────
        elapsed_total = time() - t_start
        log_msg("  $update_folder done in $(fmt(elapsed_total)) — $n_received / $n_total completed")

        # ── Deduplicate failed_buildings_<update>.csv ─────────────────
        if isfile(failed_csv)
            fail_df = CSV.read(failed_csv, DataFrame; header = 1, normalizenames = true)
            unique!(fail_df, :bldg_id)
            CSV.write(failed_csv, fail_df)
        end
    end

    log_msg("Case '$case_name' — all update folders complete")
end

# ── Final summary ─────────────────────────────────────────────────────────
log_msg("═"^70)
log_msg("ALL CASES COMPLETE")
log_msg("═"^70)
for case_name in all_cases
    case_dir       = joinpath(STUDY_DIR, case_name)
    update_folders = discover_update_folders(case_dir)
    for update_folder in update_folders
        done = count(
            d -> isfile(joinpath(case_dir, d, update_folder, "out", "ts.csv")),
            readdir(case_dir)
        )
        log_msg("  $case_name / $update_folder: $done completed")
    end
end
log_msg("Log file: $LOG_FILE")
