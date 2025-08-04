def tsp_data_generator(batch_size, problem_size, gen_type=None, uniform_ratio=0, gen_weights=[17, 19, 12]):
    import torch
    import random
    from evolved_gen.gpt_R import generate_r_type
    from evovled_gen.gpt_C import generate_c_type
    from evolved_gen.gpt_RC import generate_rc_type
    
    assert 0 <= uniform_ratio <= 1, "uniform_ratio must be in [0, 1]"
    type_choices = ["S3", "S1", "S2"]
    if gen_type is None:
        gen_type = random.choices(type_choices, weights=gen_weights, k=1)[0]



    num_uniform = int(problem_size * uniform_ratio)
    num_structured = problem_size - num_uniform

    uniform_part = torch.rand(batch_size, num_uniform, 2)

    # Generate the structured part
    if gen_type == "s3":
        structured_part = generate_c_type(batch_size=batch_size, problem_size=num_structured)
    elif gen_type == "S1":
        structured_part = generate_r_type(batch_size=batch_size, problem_size=num_structured)
    elif gen_type == "S2":
        structured_part = generate_rc_type(batch_size=batch_size, problem_size=num_structured)


    # Assemble uniform + structured and randomly shuffle the order
    combined = torch.cat([uniform_part, structured_part], dim=1)
    perm = torch.argsort(torch.rand(batch_size, problem_size), dim=1)
    shuffled = torch.stack([
        combined[i][perm[i]] for i in range(batch_size)
    ])


    return shuffled  # shape: (batch_size, problem_size, 2)