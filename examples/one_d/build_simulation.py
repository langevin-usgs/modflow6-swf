import matplotlib.pyplot as plt
import numpy as np
import flopy


def build_simulation(
    sim_ws,
    exe_name,
    nreach,
    dx,
    h0,
    h1,
    roughness,
    dev_swr_conductance,
    central_in_space,
    save_velocity,
):
    ncol = nreach
    nper = 1
    perlen = nper * [1036800.0]
    nstp = nper * [1]
    tsmult = nper * [1]

    tdis_rc = []
    for i in range(nper):
        tdis_rc.append((perlen[i], nstp[i], tsmult[i]))

    name = "swf"

    # build MODFLOW 6 files
    sim = flopy.mf6.MFSimulation(
        sim_name=f"{name}_sim",
        version="mf6",
        exe_name=exe_name,
        sim_ws=sim_ws,
    )

    # create tdis package
    ats_filerecord = None
    _ = flopy.mf6.ModflowTdis(
        sim,
        ats_filerecord=ats_filerecord,
        time_units="SECONDS",
        nper=nper,
        perioddata=tdis_rc,
    )

    def add_model(sim, modelname, dis_type):
        # surface water model
        swf = flopy.mf6.ModflowSwf(sim, modelname=modelname, save_flows=True)

        nouter, ninner = 200, 30
        hclose, relax = 1e-8, 0.97
        imsswf = flopy.mf6.ModflowIms(
            sim,
            print_option="SUMMARY",
            outer_dvclose=hclose,
            outer_maximum=nouter,
            under_relaxation="DBD",
            under_relaxation_theta=0.9,
            under_relaxation_kappa=0.0001,
            under_relaxation_gamma=0.0,
            inner_maximum=ninner,
            inner_dvclose=hclose,
            linear_acceleration="BICGSTAB",
            scaling_method="NONE",
            reordering_method="NONE",
            relaxation_factor=relax,
            preconditioner_levels=7,
            # backtracking_number=5,
            # backtracking_tolerance=1.0,
            # backtracking_reduction_factor=0.3,
            # backtracking_residual_limit=100.0,
            filename=f"{modelname}.ims",
        )
        sim.register_ims_package(imsswf, [swf.name])

        land_surface = 0.0
        if dis_type == "dis2d":
            _ = flopy.mf6.ModflowSwfdis2D(
                swf,
                nrow=1,
                ncol=ncol,
                delr=dx,
                delc=1.0,
                botm=land_surface,
                xorigin=0.0,
            )
        elif dis_type == "disv1d":
            nodes = ncol
            nvert = nodes + 1
            vertices = [[j, j * dx, 0.0] for j in range(nodes + 1)]
            cell2d = [[j, 0.5, 2, j, j + 1] for j in range(nodes)]
            _ = flopy.mf6.ModflowSwfdisv1D(
                swf,
                nodes=nodes,
                nvert=nvert,
                length=dx,
                width=1.0,
                bottom=land_surface,
                idomain=1,
                vertices=vertices,
                cell2d=cell2d,
            )
        else:
            raise Exception(f"unknown dis type: {dis_type}")

        _ = flopy.mf6.ModflowSwfdfw(
            swf,
            dev_swr_conductance=dev_swr_conductance,
            central_in_space=central_in_space,
            print_flows=False,
            save_flows=True,
            save_velocity=save_velocity,
            manningsn=roughness,
            idcxs=None,
        )

        _ = flopy.mf6.ModflowSwfic(
            swf,
            strt=0.5 * (h0 + h1),
        )

        # output control
        _ = flopy.mf6.ModflowSwfoc(
            swf,
            budget_filerecord=f"{modelname}.bud",
            stage_filerecord=f"{modelname}.stage",
            saverecord=[
                ("STAGE", "ALL"),
                ("BUDGET", "ALL"),
            ],
            printrecord=[
                ("BUDGET", "ALL"),
            ],
        )

        # Assign constant heads in first and last reach.
        if dis_type == "dis2d":
            spd = [(0, 0, h0), (0, ncol - 1, h1)]
        elif dis_type == "disv1d":
            spd = [(0, h0), (nodes - 1, h1)]
        _ = flopy.mf6.ModflowSwfchd(
            swf,
            maxbound=len(spd),
            print_input=True,
            print_flows=True,
            stress_period_data=spd,
        )

    swfname = "overland"
    add_model(sim, swfname, dis_type="dis2d")

    swfname = "channel"
    add_model(sim, swfname, dis_type="disv1d")

    return sim


def make_plot(sim, x, h_analytical_solution, extent, istep, symbols=None):
    if symbols is None:
        symbols = {}
    sim_ws = sim.simulation_data.mfpath.get_sim_path()

    # setup channel
    fpth = sim_ws / "channel.stage"
    stage_channel = None
    if fpth.exists():
        stage_obj = flopy.utils.HeadFile(fpth, text="STAGE")
        stage_channel = stage_obj.get_data().flatten()
        if "channel" in symbols:
            channel_symbol = symbols["channel"]
        else:
            channel_symbol = {
                "marker": None,
                "mfc": "none",
                "color": "blue",
                "ls": "dashed",
                "linewidth": 1.0,
                "label": "Channel (DISV1D)",
            }
        if "channel_error" in symbols:
            channel_error_symbol = symbols["channel_error"]
            channel_error_symbol["ls"] = "solid"
        else:
            channel_error_symbol = channel_symbol

    fpth = sim_ws / "overland.stage"
    stage_overland = None
    if fpth.exists():
        stage_obj = flopy.utils.HeadFile(fpth, text="STAGE")
        stage_overland = stage_obj.get_data().flatten()
        if "overland" in symbols:
            overland_symbol = symbols["overland"]
        else:
            overland_symbol = {
                "marker": None,
                "mfc": "none",
                "color": "red",
                "ls": "dotted",
                "linewidth": 1.0,
                "label": "Overland (DIS2D)",
            }
        if "overland_error" in symbols:
            overland_error_symbol = symbols["overland_error"]
            overland_error_symbol["ls"] = "solid"
        else:
            overland_error_symbol = overland_symbol

    if "analytical" in symbols:
        analytical_symbol = symbols["analytical"]
    else:
        analytical_symbol = {
            "marker": None,
            "mfc": "none",
            "color": "black",
            "ls": "solid",
            "linewidth": 2.0,
            "label": "Analytical",
        }

    with flopy.plot.styles.USGSPlot():

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(2, 1, 1)
        line_set = ax.plot(x, h_analytical_solution, **analytical_symbol)
        line_sets = line_set
        if stage_channel is not None:
            xp = x[::istep]
            line_set = ax.plot(
                xp,
                stage_channel[::istep],
                **channel_symbol,
            )
            line_sets.extend(line_set)
        if stage_overland is not None:
            xp = x[::istep]
            line_set = ax.plot(
                xp,
                stage_overland[::istep],
                **overland_symbol,
            )
            line_sets.extend(line_set)
        xmin, xmax, ymin, ymax = extent
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel("Stage, in meters")
        ax.legend(loc="lower left")

        # make error figure
        ax = fig.add_subplot(2, 1, 2)
        if stage_channel is not None:
            diff = stage_channel - h_analytical_solution
            channel_error_symbol["label"] = f"Error Channel (abs max={np.abs(diff).max():.4f})"
            line_set = ax.plot(
                x, diff, **channel_error_symbol
            )
            line_sets.extend(line_set)
        if stage_overland is not None:
            diff = stage_overland - h_analytical_solution
            overland_error_symbol["label"] = f"Error Overland (abs max={np.abs(diff).max():.4f})"
            line_set = ax.plot(
                x, diff, **overland_error_symbol
            )
            line_sets.extend(line_set)
        labs = [line_set.get_label() for line_set in line_sets]
        # ax.legend(line_sets, labs, loc="lower left")
        ax.legend(loc="lower left")

        ax.set_ylim(-1, 1)
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("X-location, in meters")
        ax.set_ylabel("Error, in meters")



    return fig, ax