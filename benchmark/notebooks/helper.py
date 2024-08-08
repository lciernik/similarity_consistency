def get_model_ids(fn):
    with open(fn, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    return lines