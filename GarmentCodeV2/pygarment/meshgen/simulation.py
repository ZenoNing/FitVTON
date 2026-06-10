# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Cloth
#
# Shows a simulation of an FEM cloth model colliding against a static
# rigid body mesh using the wp.sim.ModelBuilder().
#
###########################################################################

import sys
import time
import traceback
import platform
import multiprocessing
import signal
import trimesh
import numpy as np
from pathlib import Path

# Warp
import warp as wp

# Custom code
from pygarment.meshgen.render.pythonrender import render_images, render_images_for_frame, render_multi_images
from pygarment.meshgen.garment import Cloth, MultiCloth
from pygarment.meshgen.sim_config import SimConfig, PathCofig

wp.init()

_CURRENT_WP_DEVICE = None


def set_wp_device_once(gpu_id):
    """Avoid repeated Warp device setup when batching samples on one GPU."""
    global _CURRENT_WP_DEVICE
    target_device = f'cuda:{gpu_id}'
    if _CURRENT_WP_DEVICE != target_device:
        wp.set_device(target_device)
        _CURRENT_WP_DEVICE = target_device

class SimulationError(BaseException):
    """To be rised when panel stitching cannot be executed correctly"""
    pass

class FrameTimeOutError(BaseException):
    """To be rised when frame takes too long to simulate"""
    pass

class SimTimeOutError(BaseException):
    """To be rised when simulation takes too long"""
    pass

def optimize_garment_storage(paths: PathCofig):
    """Prepare the data element for compact storage: store the meshes as ply instead of obj, 
        remove texture files 
    """
    # Objs to ply
    try:
        boxmesh = trimesh.load(paths.g_box_mesh)
        boxmesh.export(paths.g_box_mesh_compressed)
        paths.g_box_mesh.unlink()
    except BaseException:
        pass

    try:
        simmesh = trimesh.load(paths.g_sim)
        simmesh.export(paths.g_sim_compressed)
        paths.g_sim.unlink()
    except BaseException:
        pass

    # Remove large texture file and mtl -- not so necessary
    paths.g_texture_fabric.unlink(missing_ok=True)
    paths.g_mtl.unlink(missing_ok=True)


def update_progress(progress, total):
    """Progress bar in console"""
    # https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    amtDone = progress / total
    num_dash = int(amtDone * 50)
    sys.stdout.write('\rProgress: [{0:50s}] {1:.1f}%'.format('#' * num_dash + '-' * (50 - num_dash), amtDone * 100))
    sys.stdout.flush()


def sim_frame_sequence(
        garment,
        config,
        store_usd=False,
        verbose=False,
        dynamic_frames=None,
        render_config=None,
        render_each_frame=False,
        render_stride=1,
        render_each_frame_include_masks=False,
        collect_frame_verts=False,
        render_multi_output_dir=None,
        render_multi_name=None,
    ):

    frame_timeout_after = config.max_frame_time
    collected_frames = []  # list of (cloth_verts, body_verts) numpy arrays
    # Save initial state
    if store_usd:
        garment.render_usd_frame()

    start_time = time.time()
    for frame in range(0, config.max_sim_steps):
        
        if verbose:
            print(f'\n------ Frame {frame + 1} ------')
        else:
            update_progress(frame, config.max_sim_steps)

        garment.frame = frame 

        #Run frame and raise FrameTimeOutError if frame takes too long to simulate

        static = False
        if frame == 0:
            frame_timeout_after *= 2
        try:
            if platform.system() == "Windows":
                """https://stackoverflow.com/a/14920854"""

                if frame == 0: #only do it on first frame due to slowdown
                    p_frame = multiprocessing.Process(target=garment.run_frame(), name="FrameSimulation")
                    p_frame.start()

                    # Wait timeout_after seconds for garment.run_frame()
                    p_frame.join(frame_timeout_after)

                    # If thread is active
                    if p_frame.is_alive():
                        # Terminate the process
                        p_frame.terminate()
                        p_frame.join()
                        raise TimeoutError
                else:
                    garment.run_frame()

            elif platform.system() in ["Linux", "OSX"]:
                """https://code-maven.com/python-timeout"""
                import threading
                if threading.current_thread() is threading.main_thread():
                    def alarm_handler(signum, frame):
                        raise TimeoutError

                    signal.signal(signal.SIGALRM, alarm_handler)
                    signal.alarm(frame_timeout_after)
                    s_time = time.time()
                    try:
                        garment.run_frame()
                    except TimeoutError as ex:
                        raise TimeoutError
                    else:
                        e_time = time.time() - s_time
                        signal.alarm(0)
                else:
                    # signal.alarm not available outside main thread — run without timeout
                    garment.run_frame()
        except TimeoutError as e:
            raise FrameTimeOutError

        # Optional per-frame render (PNG sequence). This is significantly slower than sim.
        if render_each_frame and render_config is not None and (frame % max(int(render_stride), 1) == 0):
            # Update the on-disk mesh that pythonrender reads from (paths.g_sim)
            garment.save_frame(save_v_norms=False)
            if hasattr(garment, 'garment_paths'):
                frame_dir = Path(render_multi_output_dir if render_multi_output_dir is not None else garment.paths.out_el) / 'frames'
                frame_name = f"{render_multi_name or garment.name}_frame_{frame:05d}"
                render_multi_images(
                    garment.garment_paths,
                    garment.v_body,
                    garment.f_body,
                    render_config,
                    frame_dir,
                    name=frame_name,
                    include_masks=render_each_frame_include_masks,
                )
            else:
                render_images_for_frame(
                    garment.paths,
                    garment.v_body,
                    garment.f_body,
                    render_config,
                    frame_idx=frame,
                    include_masks=render_each_frame_include_masks,
                )

        # Collect vertex snapshots in memory (much faster than per-frame rendering)
        if collect_frame_verts and (frame % max(int(render_stride), 1) == 0):
            collected_frames.append((
                garment.current_verts.copy(),
                garment.v_body.copy() if garment.v_body is not None else None,
            ))

        if verbose:
            num_cloth_cloth_contacts = garment.count_self_intersections()
            print(f'\nSelf-Intersection: {num_cloth_cloth_contacts}')

        if (garment.static_detected_frame is None and 
            frame >= config.zero_gravity_steps and 
            frame >= config.min_sim_steps
        ):
            static, _ = garment.is_static()
            if static and garment.static_detected_frame is None:
                garment.static_detected_frame = frame

        if garment.static_detected_frame is not None:
            if dynamic_frames is None:
                break

            dynamic_end_frame = garment.static_detected_frame + dynamic_frames - 1
            if frame < dynamic_end_frame:
                garment.dynamic_body_simulation = True
            else:
                garment.dynamic_body_simulation = False
                post_dynamic_frames = getattr(config, 'post_dynamic_frames', 100)
                if post_dynamic_frames >= 0:
                    if frame >= dynamic_end_frame + post_dynamic_frames:
                        break
                else:
                    static, _ = garment.is_static()
                    if static:
                        break
                    break

        runtime = time.time() - start_time
        if runtime > config.max_sim_time:
            raise SimTimeOutError

    return collected_frames
        

def run_sim(
        cloth_name, props, paths: PathCofig, 
        save_v_norms=False, store_usd=False, 
        optimize_storage=False,
    verbose=False,
    render_each_frame=False,
    render_stride=1,
    render_each_frame_include_masks=False): 
    """Initialize and run the simulation
    !! Important !! 
        'store_usd' parameter slows down the simulation to CPU rates because of required CPU-GPU copies and file writes. Use only for debugging
    """
    sim_props = props['sim']
    render_props = props['render']

    start_time = time.time()

    config = SimConfig(sim_props['config'])   # Why separate class at all? 
    garment = Cloth(cloth_name, config, paths, caching=store_usd)

    try:
        print("Simulation..")
        sim_frame_sequence(
            garment,
            config,
            store_usd,
            verbose=verbose,
            render_config=render_props['config'] if 'config' in render_props else None,
            render_each_frame=render_each_frame,
            render_stride=render_stride,
            render_each_frame_include_masks=render_each_frame_include_masks,
        )
    
    except FrameTimeOutError:
        print(f"FrameTimeOutError at frame {garment.frame}")
        props.add_fail('sim', 'frame_timeout', cloth_name)
    except SimTimeOutError:
        print("SimTimeOutError")
        props.add_fail('sim', 'simulation_timeout', cloth_name)
    except SimulationError:
        print("Simulation failed")
        props.add_fail('sim', 'gt_edges_creation', cloth_name)
    except BaseException as e:
        print(f'Sim::{cloth_name}::crashed with {e}')

        if isinstance(e, KeyboardInterrupt):
            # Allow to stop simulation loops by keyboard interrupt
            # It's not a real crash, so don't write down the failure
            sec = round(time.time() - start_time, 3)
            min = int(sec / 60)
            print(f"Simulation pipeline took: {min} m {sec - min * 60} s")
            raise e

        traceback.print_exc()
        props.add_fail('sim', 'crashes', cloth_name)
    else:  # Other quality checks
        if garment.frame == config.max_sim_steps - 1:
            _, non_st_count = garment.is_static()
            print('\nFailed to achieve static equilibrium for {} with {} non-static vertices out of {}'.format(
                cloth_name, non_st_count, len(garment.current_verts)))
            props.add_fail('sim', 'static_equilibrium', cloth_name)

        if time.time() - start_time < 0.5:  # 0.5 sec  -- finished suspiciously fast
            props.add_fail('sim', 'fast_finish', cloth_name)

        # 3D penetrations
        num_body_collisions = garment.count_body_intersections()
        print("BODY CLOTH INTERSECTIONS: ", num_body_collisions)
        num_self_collisions = garment.count_self_intersections()

        sim_props['stats']['body_collisions'][cloth_name] = num_body_collisions
        sim_props['stats']['self_collisions'][cloth_name] = num_self_collisions

        if num_body_collisions > config.max_body_collisions:
            props.add_fail('sim', 'cloth_body_intersection', cloth_name)
        if num_self_collisions: 
            print(f'Self-Intersecting with {num_self_collisions}, '
                  f'is fail: {num_self_collisions > config.max_self_collisions}')
            if num_self_collisions > config.max_self_collisions:
                props.add_fail('sim', 'cloth_self_intersection', cloth_name)
        else:
            print('Not self-intersecting!!!')

    # ---- Postprocessing ----
    # NOTE: Attempt even on failures for accurate picture and post-analysis
    frame = garment.frame
    print(f"\nSimulation took #frames={frame + 1}")

    sim_props['stats']['sim_time'][cloth_name] = sim_time = time.time() - start_time
    sim_props['stats']['spf'][cloth_name] = sim_time / frame if frame else sim_time
    sim_props['stats']['fin_frame'][cloth_name] = frame

    garment.save_frame(save_v_norms=save_v_norms) #saving after stats

    # Render images
    s_time = time.time()
    render_images(paths, garment.v_body, garment.f_body, render_props['config'])
    render_image_time = time.time() - s_time
    render_props['stats']['render_time'][cloth_name] = render_image_time  
    print(f"Rendering {cloth_name} took {render_image_time}s")

    if optimize_storage:
        optimize_garment_storage(paths)

    # Final info output
    sec = round(time.time() - start_time, 3)
    min = int(sec / 60)
    print(f"\nSimulation pipeline took: {min} m {sec - min * 60} s")


def run_sim_new(
        cloth_name, props, paths: PathCofig, 
        body_sequence=None, body_faces=None,
        save_v_norms=False, store_usd=False, 
        optimize_storage=False,
        verbose=False,
        dynamic_frames=None,  # Use the length of the body sequence
        gpu_id=0,
        render_each_frame=False,
        render_stride=1,
        render_each_frame_include_masks=False,
        collect_frame_verts=False): 
    """Initialize and run the simulation
    !! Important !! 
        'store_usd' parameter slows down the simulation to CPU rates because of required CPU-GPU copies and file writes. Use only for debugging
    """

    set_wp_device_once(gpu_id)
    sim_props = props['sim']
    render_props = props['render']

    start_time = time.time()

    config = SimConfig(sim_props['config'])   # Why separate class at all? 
    garment = Cloth(cloth_name, config, paths, caching=store_usd, body_sequence=body_sequence, body_faces=body_faces)
    collected_frames = []

    try:
        print("Simulation..")
        collected_frames = sim_frame_sequence(
            garment,
            config,
            store_usd,
            verbose=verbose,
            dynamic_frames=dynamic_frames,
            render_config=render_props['config'] if 'config' in render_props else None,
            render_each_frame=render_each_frame,
            render_stride=render_stride,
            render_each_frame_include_masks=render_each_frame_include_masks,
            collect_frame_verts=collect_frame_verts,
        )

    except FrameTimeOutError:
        print(f"FrameTimeOutError at frame {garment.frame}")
        props.add_fail('sim', 'frame_timeout', cloth_name)
    except SimTimeOutError:
        print("SimTimeOutError")
        props.add_fail('sim', 'simulation_timeout', cloth_name)
    except SimulationError:
        print("Simulation failed")
        props.add_fail('sim', 'gt_edges_creation', cloth_name)
    except BaseException as e:
        print(f'Sim::{cloth_name}::crashed with {e}')

        if isinstance(e, KeyboardInterrupt):
            # Allow to stop simulation loops by keyboard interrupt
            # It's not a real crash, so don't write down the failure
            sec = round(time.time() - start_time, 3)
            min = int(sec / 60)
            print(f"Simulation pipeline took: {min} m {sec - min * 60} s")
            raise e

        traceback.print_exc()
        props.add_fail('sim', 'crashes', cloth_name)
    else:  # Other quality checks
        if garment.frame == config.max_sim_steps - 1:
            _, non_st_count = garment.is_static()
            print('\nFailed to achieve static equilibrium for {} with {} non-static vertices out of {}'.format(
                cloth_name, non_st_count, len(garment.current_verts)))
            props.add_fail('sim', 'static_equilibrium', cloth_name)

        if time.time() - start_time < 0.5:  # 0.5 sec  -- finished suspiciously fast
            props.add_fail('sim', 'fast_finish', cloth_name)

        # 3D penetrations
        num_body_collisions = garment.count_body_intersections()
        print("BODY CLOTH INTERSECTIONS: ", num_body_collisions)
        num_self_collisions = garment.count_self_intersections()

        sim_props['stats']['body_collisions'][cloth_name] = num_body_collisions
        sim_props['stats']['self_collisions'][cloth_name] = num_self_collisions

        if num_body_collisions > config.max_body_collisions:
            props.add_fail('sim', 'cloth_body_intersection', cloth_name)
        if num_self_collisions: 
            print(f'Self-Intersecting with {num_self_collisions}, '
                  f'is fail: {num_self_collisions > config.max_self_collisions}')
            if num_self_collisions > config.max_self_collisions:
                props.add_fail('sim', 'cloth_self_intersection', cloth_name)
        else:
            print('Not self-intersecting!!!')

    # ---- Postprocessing ----
    # NOTE: Attempt even on failures for accurate picture and post-analysis
    frame = garment.frame
    print(f"\nSimulation took #frames={frame + 1}")
    sim_props['stats']['sim_time'][cloth_name] = sim_time = time.time() - start_time
    sim_props['stats']['spf'][cloth_name] = sim_time / frame if frame else sim_time
    sim_props['stats']['fin_frame'][cloth_name] = frame

    garment.save_frame(save_v_norms=save_v_norms) #saving after stats

    # Render images
    s_time = time.time()
    render_images(paths, garment.v_body, garment.f_body, render_props['config'])
    render_image_time = time.time() - s_time
    render_props['stats']['render_time'][cloth_name] = render_image_time  
    print(f"Rendering {cloth_name} took {render_image_time}s")

    if optimize_storage:
        optimize_garment_storage(paths)

    # Final info output
    sec = round(time.time() - start_time, 3)
    min = int(sec / 60)
    print(f"\nSimulation pipeline took: {min} m {sec - min * 60} s")

    if collect_frame_verts:
        return collected_frames


def settle_frame_sequence(garment, config, verbose=False):
    """Run a garment until static equilibrium without advancing the body sequence."""
    frame_timeout_after = config.max_frame_time
    start_time = time.time()

    for frame in range(0, config.max_sim_steps):
        if verbose:
            print(f'\n------ Stage-1 Frame {frame + 1} ------')
        else:
            update_progress(frame, config.max_sim_steps)

        garment.frame = frame
        if frame == 0:
            frame_timeout_after *= 2

        try:
            garment.run_frame()
        except TimeoutError:
            raise FrameTimeOutError

        if (garment.static_detected_frame is None and
                frame >= config.zero_gravity_steps and
                frame >= config.min_sim_steps):
            static, _ = garment.is_static()
            if static:
                garment.static_detected_frame = frame
                break

        runtime = time.time() - start_time
        if runtime > config.max_sim_time:
            raise SimTimeOutError

    return garment.current_verts.copy()


def run_sim_multi_new(
        cloth_names,
        props,
        paths_list,
        first_garment_index,
        body_sequence=None,
        body_faces=None,
        save_v_norms=False,
        store_usd=False,
        optimize_storage=False,
        verbose=False,
        dynamic_frames=None,
        gpu_id=0,
        combined_render_dir=None,
        combined_render_name='multi',
        waistband_side_mode=None,
        render_each_frame=False,
        render_stride=1,
        render_each_frame_include_masks=False):
    """Two-stage multi-garment simulation with cloth-cloth collision.

    Stage 1 simulates only the first garment until it settles. Stage 2 builds a
    single Warp model containing both garments, initializes the first garment
    from the settled result, and simulates both garments together so XPBD
    resolves their cloth-cloth contacts.
    """

    set_wp_device_once(gpu_id)
    sim_props = props['sim']
    render_props = props['render']
    config = SimConfig(sim_props['config'])
    start_time = time.time()

    first_name = cloth_names[first_garment_index]
    first_paths = paths_list[first_garment_index]

    print(f"Stage 1: settling {first_name} alone")
    stage1 = MultiCloth(
        f'{first_name}_stage1',
        config,
        [first_paths],
        garment_names=[first_name],
        caching=store_usd,
        body_sequence=body_sequence,
        body_faces=body_faces,
    )
    settled_first_vertices = settle_frame_sequence(stage1, config, verbose=verbose)

    print(f"\nStage 2: simulating {', '.join(cloth_names)} together")
    initial_vertices = {first_name: settled_first_vertices}
    garment = MultiCloth(
        '_'.join(cloth_names),
        config,
        paths_list,
        garment_names=cloth_names,
        caching=store_usd,
        body_sequence=body_sequence,
        body_faces=body_faces,
        initial_vertices=initial_vertices,
        waistband_side_mode=waistband_side_mode,
    )

    try:
        print("Simulation..")
        sim_frame_sequence(
            garment,
            config,
            store_usd,
            verbose=verbose,
            dynamic_frames=dynamic_frames,
            render_config=render_props['config'] if 'config' in render_props else None,
            render_each_frame=render_each_frame,
            render_stride=render_stride,
            render_each_frame_include_masks=render_each_frame_include_masks,
            collect_frame_verts=False,
            render_multi_output_dir=combined_render_dir,
            render_multi_name=combined_render_name,
        )

    except FrameTimeOutError:
        print(f"FrameTimeOutError at frame {garment.frame}")
        props.add_fail('sim', 'frame_timeout', '_'.join(cloth_names))
    except SimTimeOutError:
        print("SimTimeOutError")
        props.add_fail('sim', 'simulation_timeout', '_'.join(cloth_names))
    except SimulationError:
        print("Simulation failed")
        props.add_fail('sim', 'gt_edges_creation', '_'.join(cloth_names))
    except BaseException as e:
        print(f'Sim::{"_".join(cloth_names)}::crashed with {e}')
        if isinstance(e, KeyboardInterrupt):
            sec = round(time.time() - start_time, 3)
            min = int(sec / 60)
            print(f"Simulation pipeline took: {min} m {sec - min * 60} s")
            raise e
        traceback.print_exc()
        props.add_fail('sim', 'crashes', '_'.join(cloth_names))
    else:
        if garment.frame == config.max_sim_steps - 1:
            _, non_st_count = garment.is_static()
            print('\nFailed to achieve static equilibrium for {} with {} non-static vertices out of {}'.format(
                '_'.join(cloth_names), non_st_count, len(garment.current_verts)))
            props.add_fail('sim', 'static_equilibrium', '_'.join(cloth_names))

        num_body_collisions = garment.count_body_intersections()
        print("BODY CLOTH INTERSECTIONS: ", num_body_collisions)
        num_self_collisions = garment.count_self_intersections()
        sim_props['stats']['body_collisions']['_'.join(cloth_names)] = num_body_collisions
        sim_props['stats']['self_collisions']['_'.join(cloth_names)] = num_self_collisions
        if num_body_collisions > config.max_body_collisions:
            props.add_fail('sim', 'cloth_body_intersection', '_'.join(cloth_names))
        if num_self_collisions:
            print(f'Self-Intersecting with {num_self_collisions}, '
                  f'is fail: {num_self_collisions > config.max_self_collisions}')
            if num_self_collisions > config.max_self_collisions:
                props.add_fail('sim', 'cloth_self_intersection', '_'.join(cloth_names))
        else:
            print('Not self-intersecting!!!')

    frame = garment.frame
    print(f"\nSimulation took #frames={frame + 1}")
    sim_time = time.time() - start_time
    sim_props['stats']['sim_time']['_'.join(cloth_names)] = sim_time
    sim_props['stats']['spf']['_'.join(cloth_names)] = sim_time / frame if frame else sim_time
    sim_props['stats']['fin_frame']['_'.join(cloth_names)] = frame

    garment.save_frame(save_v_norms=save_v_norms)

    s_time = time.time()
    if combined_render_dir is None:
        combined_render_dir = paths_list[0].out_el
    render_multi_images(paths_list, garment.v_body, garment.f_body, render_props['config'],
                        combined_render_dir, name=combined_render_name)
    render_image_time = time.time() - s_time
    render_props['stats']['render_time']['_'.join(cloth_names)] = render_image_time
    print(f"Rendering {'_'.join(cloth_names)} took {render_image_time}s")

    if optimize_storage:
        for paths in paths_list:
            optimize_garment_storage(paths)

    sec = round(time.time() - start_time, 3)
    min = int(sec / 60)
    print(f"\nSimulation pipeline took: {min} m {sec - min * 60} s")

    return garment
