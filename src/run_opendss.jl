"""
OpenDSS Study Runner
─────────────────────────────────────────────────────────────────────────────
Runs yearly (8760-hour) power flow simulations for each DECARB case in a study.

The network model (Master.dss, line/transformer definitions) lives in the
template directory and is never copied. Per-case loadshape CSVs live in the
study's opendss/{case}/ directory and are loaded at runtime.

Usage:
    julia --project run_opendss.jl [study_num]

If study_num is omitted, uses the latest study.
─────────────────────────────────────────────────────────────────────────────
"""

# Handle interactive REPL relaunch (OpenDSSDirect init crash workaround)
if isinteractive()
    println("Interactive session — relaunching as subprocess...")
    flush(stdout)
    julia_exe = joinpath(Sys.BINDIR, Base.julia_exename())
    project_dir = dirname(@__DIR__)  # project root (one level up from src/)
    run(`$julia_exe --project=$project_dir --warn-overwrite=no $(@__FILE__) $(ARGS...)`)
    return
end

using OpenDSSDirect
using Statistics
using CSV
using DataFrames
using Dates

# ── Settings ──────────────────────────────────────────────────────────────
const PROJECT_ROOT = dirname(@__DIR__)
const TEMPLATE_DIR = joinpath(PROJECT_ROOT, "in", "opendss_input_templates")
const STUDIES_ROOT = joinpath(PROJECT_ROOT, "out", "studies")
const ZONE_NAME   = "P100U"
const NUM_STEPS   = 8760
const STEPSIZE    = "1h"

# ── Find study directory ──────────────────────────────────────────────────
function find_study(study_num=nothing)
    if study_num !== nothing
        d = joinpath(STUDIES_ROOT, "study_$study_num")
        isdir(d) || error("Study not found: $d")
        return d
    end
    # Find latest
    dirs = filter(d -> startswith(d, "study_") && isdir(joinpath(STUDIES_ROOT, d)),
                  readdir(STUDIES_ROOT))
    isempty(dirs) && error("No studies found in $STUDIES_ROOT")
    nums = [parse(Int, split(d, "_")[2]) for d in dirs]
    return joinpath(STUDIES_ROOT, "study_$(maximum(nums))")
end

# ── Discover feeders in the template ──────────────────────────────────────
function discover_feeders(zone_template_dir)
    tasks = NamedTuple[]
    for sub in sort(readdir(zone_template_dir))
        sub_path = joinpath(zone_template_dir, sub)
        isdir(sub_path) || continue
        for feeder in sort(readdir(sub_path))
            feeder_dir = joinpath(sub_path, feeder)
            isdir(feeder_dir) || continue
            master = joinpath(feeder_dir, "Master.dss")
            isfile(master) || continue
            push!(tasks, (name=feeder, master=master, dir=feeder_dir, substation=sub))
        end
    end
    return tasks
end

# ── Compile Master.dss from template ──────────────────────────────────────
function compile_feeder(master_path, feeder_dir, loadshape_dir)
    OpenDSSDirect.Text.Command("Clear")

    # First, compile from the template directory (topology, lines, xfmrs)
    cd(feeder_dir)
    lines = readlines(master_path)
    for line in lines
        s = strip(line)
        isempty(s) && continue
        low = lowercase(s)
        # Skip solve/export/plot commands
        (startswith(low, "solve") || startswith(low, "export") || startswith(low, "plot ")) && continue
        OpenDSSDirect.Text.Command(String(s))
    end

    # Now redirect loadshapes to the case-specific kW and kvar files
    if isdir(loadshape_dir)
        n_loads = OpenDSSDirect.Loads.Count()
        if n_loads > 0
            OpenDSSDirect.Loads.First()
            for _ in 1:n_loads
                load_name = OpenDSSDirect.Loads.Name()
                kw_path = joinpath(loadshape_dir, "res_kw_$(load_name).csv")
                kvar_path = joinpath(loadshape_dir, "res_kvar_$(load_name).csv")
                if isfile(kw_path)
                    shape_name = "shape_$(load_name)"
                    # Use useactual=true with mult/qmult file references
                    kw_posix = replace(kw_path, "\\" => "/")
                    cmd = "New Loadshape.$shape_name npts=$NUM_STEPS interval=1 useactual=true mult=(file=$kw_posix)"
                    if isfile(kvar_path)
                        kvar_posix = replace(kvar_path, "\\" => "/")
                        cmd *= " qmult=(file=$kvar_posix)"
                    end
                    OpenDSSDirect.Text.Command(cmd)
                    OpenDSSDirect.Text.Command("Load.$load_name.yearly=$shape_name")
                end
                OpenDSSDirect.Loads.Next()
            end
        end
    end
end

# ── DSS helpers ───────────────────────────────────────────────────────────
function get_all_bus_voltages_pu()
    vmags = Float64[]
    for bus in OpenDSSDirect.Circuit.AllBusNames()
        OpenDSSDirect.Circuit.SetActiveBus(bus)
        pu = OpenDSSDirect.Bus.puVmagAngle()
        n = OpenDSSDirect.Bus.NumNodes()
        for i in 1:n
            v = pu[2i - 1]
            v > 0.01 && push!(vmags, v)
        end
    end
    return vmags
end

function get_head_line()
    OpenDSSDirect.Lines.First()
    all_lines = String[]
    while true
        push!(all_lines, OpenDSSDirect.Lines.Name())
        OpenDSSDirect.Lines.Next() == 0 && break
    end
    head = filter(l -> occursin("p100uhs", lowercase(l)), all_lines)
    return isempty(head) ? all_lines[1] : head[1]
end

function get_line_loading(line_name)
    OpenDSSDirect.Circuit.SetActiveElement("Line.$line_name")
    cur = OpenDSSDirect.CktElement.CurrentsMagAng()
    n_cond = OpenDSSDirect.CktElement.NumConductors()
    i_max = maximum(cur[1, i] for i in 1:n_cond)
    norm = OpenDSSDirect.CktElement.NormalAmps()
    pct = norm > 0 ? (i_max / norm * 100) : 0.0
    return i_max, norm, pct
end

# ── Per-feeder simulation ─────────────────────────────────────────────────
function simulate_feeder(task, loadshape_dir, num_steps, stepsize)
    t0 = time()
    compile_feeder(task.master, task.dir, loadshape_dir)
    monitor_line = get_head_line()
    OpenDSSDirect.Text.Command("Set mode=yearly stepsize=$stepsize number=1")

    hours    = Vector{Float64}(undef, num_steps)
    v_min    = Vector{Float64}(undef, num_steps)
    v_max    = Vector{Float64}(undef, num_steps)
    v_mean   = Vector{Float64}(undef, num_steps)
    tot_kw   = Vector{Float64}(undef, num_steps)
    tot_kvar = Vector{Float64}(undef, num_steps)
    loading  = Vector{Float64}(undef, num_steps)
    head_i   = Vector{Float64}(undef, num_steps)

    for step in 1:num_steps
        OpenDSSDirect.Solution.Solve()
        hours[step] = step - 1

        vmags = get_all_bus_voltages_pu()
        v_min[step]  = isempty(vmags) ? NaN : minimum(vmags)
        v_max[step]  = isempty(vmags) ? NaN : maximum(vmags)
        v_mean[step] = isempty(vmags) ? NaN : mean(vmags)

        p = OpenDSSDirect.Circuit.TotalPower()
        tot_kw[step]   = -real(p)
        tot_kvar[step] = -imag(p)

        i_max, _, pct = get_line_loading(monitor_line)
        loading[step] = pct
        head_i[step]  = i_max
    end

    elapsed_s = round(time() - t0; digits=2)

    df = DataFrame(
        feeder           = fill(task.name, num_steps),
        substation       = fill(task.substation, num_steps),
        hour             = hours,
        v_min_pu         = v_min,
        v_max_pu         = v_max,
        v_mean_pu        = v_mean,
        total_kW         = tot_kw,
        total_kvar       = tot_kvar,
        head_loading_pct = loading,
        head_current_A   = head_i,
    )
    return df, elapsed_s
end

# ── Main ──────────────────────────────────────────────────────────────────
function main()
    # Parse study number from CLI args
    study_num = isempty(ARGS) ? nothing : parse(Int, ARGS[1])
    study_dir = find_study(study_num)
    println("Study: $study_dir")

    zone_template = joinpath(TEMPLATE_DIR, ZONE_NAME, "scenarios", "base_timeseries", "opendss")
    isdir(zone_template) || error("Zone template not found: $zone_template")

    # Discover feeders from the template
    feeders = discover_feeders(zone_template)
    println("Zone $ZONE_NAME: $(length(feeders)) feeders in template")

    # Discover cases from the study's opendss directory
    opendss_root = joinpath(study_dir, "opendss")
    isdir(opendss_root) || error("No opendss directory in study: $opendss_root")

    cases = sort(filter(d -> isdir(joinpath(opendss_root, d)), readdir(opendss_root)))
    println("Found $(length(cases)) cases: $(join(cases, ", "))")

    # Results directory inside the study
    results_root = joinpath(study_dir, "opendss_results")
    mkpath(results_root)

    # Log file
    log_file = joinpath(results_root, "run_opendss.log")
    function log_msg(msg)
        ts = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
        line = "[$ts] $msg"
        println(line)
        flush(stdout)
        open(log_file, "a") do io
            println(io, line)
            flush(io)
        end
    end

    log_msg("Starting OpenDSS runs: $(length(cases)) cases × $(length(feeders)) feeders")

    for case_name in cases
        case_results_dir = joinpath(results_root, case_name)
        zone_csv = joinpath(case_results_dir, "zone_yearly.csv")

        # Skip completed cases
        if isfile(zone_csv)
            log_msg("$case_name: already complete — skipping")
            continue
        end

        mkpath(case_results_dir)
        loadshape_dir = joinpath(opendss_root, case_name)
        log_msg("═"^60)
        log_msg("CASE: $case_name")
        log_msg("═"^60)
        log_msg("  Loadshapes: $loadshape_dir")
        log_msg("  Feeders: $(length(feeders))")

        case_results = DataFrame[]
        n_ok = 0
        n_err = 0
        t_case = time()

        for (idx, task) in enumerate(feeders)
            log_msg("  [$idx/$(length(feeders))] $(task.substation)/$(task.name) ...")

            try
                df, elapsed_s = simulate_feeder(task, loadshape_dir, NUM_STEPS, STEPSIZE)
                push!(case_results, df)
                n_ok += 1
                log_msg("    ✓ $(elapsed_s)s")
            catch e
                n_err += 1
                err_str = sprint(showerror, e)
                log_msg("    ✗ $err_str")
                # Write per-feeder error
                open(joinpath(case_results_dir, "errors.log"), "a") do io
                    println(io, "$(task.substation)/$(task.name): $err_str")
                end
            end

            cd(PROJECT_ROOT)
        end

        # Write consolidated results for this case
        if !isempty(case_results)
            zone_df = vcat(case_results...)
            CSV.write(zone_csv, zone_df)
            log_msg("Case $case_name done: $n_ok ok, $n_err errors, $(round(time()-t_case; digits=1))s")
            log_msg("  Results: $zone_csv")
        else
            log_msg("Case $case_name: no successful feeders")
        end
    end

    log_msg("═"^60)
    log_msg("ALL CASES COMPLETE")
    log_msg("═"^60)
end

main()
