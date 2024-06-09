from slurmpy import Slurm


def run_job(
        job_name,
        job_cmd,
        num_cpus=4,
        partition='gpu-2d',
        apptainer=False,
        slurm_args=None,
        log_dir='./logs',
        num_jobs_in_array=1,
):
    if apptainer:
        raise NotImplementedError("No apptainer container available for now.")
        # squash_data_dir = '/tmp'
        # copy_cmd = ["echo 'Copy data to /tmp'",
        #             "cp -r /home/space/datasets/histopathology/anomaly-detection/squash/* /tmp",
        #             "ls -l /tmp"]
        # copy_cmd = '&&'.join(copy_cmd)
        #
        # submit_cmd = f"""
        # {copy_cmd} && apptainer run --nv \
        #      -B /home/space/datasets/histopathology/anomaly-detection/colon/20x/splits:/squash/colon/20x/splits \
        #      -B /home/space/datasets/histopathology/anomaly-detection/stomach/20x/splits:/squash/stomach/20x/splits \
        #      -B /home/space/datasets/histopathology/anomaly-detection/splits:/squash/splits \
        #      -B {squash_data_dir}/colon/20x/normal/colon_HE_l01_vS007_normal_20x_80ua_images_sn_sl.sqfs:/squash/colon/20x/normal/colon_HE_l01_vS007_normal_20x_80ua_images_sn_sl:image-src=/ \
        #      -B {squash_data_dir}/colon/20x/anomaly/colon_HE_l01_vS009_anomaly_hc_20x_80ua_images_sn_sl.sqfs:/squash/colon/20x/anomaly/colon_HE_l01_vS009_anomaly_hc_20x_80ua_images_sn_sl:image-src=/ \
        #      -B {squash_data_dir}/stomach/20x/normal/stomach_normal_HE_l01_vS005_20x_80ua_images_sn_sl.sqfs:/squash/stomach/20x/normal/stomach_normal_HE_l01_vS005_20x_80ua_images_sn_sl:image-src=/ \
        #      -B {squash_data_dir}/stomach/20x/anomaly/stomach_HE_l01_vS005_anomaly_hc_20x_80ua_images_sn_sl.sqfs:/squash/stomach/20x/anomaly/stomach_HE_l01_vS005_anomaly_hc_20x_80ua_images_sn_sl:image-src=/ \
        #      -B {squash_data_dir}/oe/outlier_exposure_HE_l01_vS001_20x_images_sn_sl.sqfs:/squash/oe/outlier_exposure_HE_l01_vS001_20x_images_sn_sl:image-src=/ \
        #      -B {squash_data_dir}/oe-v2/20x/outlier_exposure_HE_l01_vS002_20x_10ua_images_sn_sl.sqfs:/squash/oe-v2/20x/outlier_exposure_HE_l01_vS002_20x_10ua_images_sn_sl:image-src=/ \
        #      -B {squash_data_dir}/oe-small-intestine/colon_mixed_HE_l01_vS007_20x_80ua_images_sn_sl.sqfs:/squash/oe-small-intestine/colon_mixed_HE_l01_vS007_20x_80ua_images_sn_sl:image-src=/ \
        #      -B {squash_data_dir}/oe-small-intestine/stomach_mixed_HE_l01_vS005_20x_80ua_images_sn_sl.sqfs:/squash/oe-small-intestine/stomach_mixed_HE_l01_vS005_20x_80ua_images_sn_sl:image-src=/ \
        #      container.sif {job_cmd}
        # """
    else:
        submit_cmd = job_cmd

    slurm_options = {
        "partition": partition,
        "cpus-per-task": num_cpus,
        "exclude": "head[026-033],head073",
        # head073 -> RuntimeError: CUDA error: no kernel image is available for execution on the device
        "nodes": 1,
        "chdir": "./",
        "output": f"{log_dir}/run_%A/%a.out",
        "error": f"{log_dir}/run_%A/%a.err",
        "array": f"0-{num_jobs_in_array - 1}" if num_jobs_in_array > 1 else "0"
    }

    if slurm_args is not None:
        for key, val in slurm_args.items():
            slurm_options[key] = val

    device, time = partition.split('-')

    if device == "gpu":
        slurm_options["gres"] = "gpu:1"

    time_mapping = {
        "test": "00-00:15:00",
        "9m": "00-00:09:00",
        "2h": "00-02:00:00",
        "5h": "00-05:00:00",
        "2d": "02-00:00:00",
        "7d": "07-00:00:00"
    }
    slurm_options["time"] = time_mapping[time]
    s = Slurm(job_name, slurm_options)
    s.run(submit_cmd)
