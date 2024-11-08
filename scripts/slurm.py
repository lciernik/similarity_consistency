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
        mem=32
):
    if apptainer:
        raise NotImplementedError("No apptainer container available for now.")
    else:
        submit_cmd = job_cmd

    if not isinstance(mem, int):
        raise TypeError("The variable mem needs to be a (positive) integer.")

    if not isinstance(mem, int):
        raise ValueError("Argument mem must be an int")

    slurm_options = {
        "partition": partition,
        "cpus-per-task": num_cpus,
        "nodes": 1,
        "chdir": "./",
        "output": f"{log_dir}/run_%A/%a.out",
        "error": f"{log_dir}/run_%A/%a.err",
        "array": f"0-{num_jobs_in_array - 1}" if num_jobs_in_array > 1 else "0",
        "mem": f"{mem}G",
    }

    if slurm_args is not None:
        for key, val in slurm_args.items():
            slurm_options[key] = val

    device, time = partition.split('-')

    if device == "gpu":
        slurm_options["gres"] = "gpu:1"
        slurm_options["constraint"] = "'[80gb|40gb|h100]'"

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
