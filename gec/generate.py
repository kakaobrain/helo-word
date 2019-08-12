import os


def generate(data_path, ckpt, system_out, ori_path=None, gen_subset=None, beam=12, max_tokens=6000, buffer_size=6000):
    """
    :param data_path: data-bin path
    :param ckpt: checkpoint path
    :param system_out: system out path to be created
    :param ori_path: bpe-tokenized ori file path (for fairseq-interactive)
    :param gen_subset: subset of the data-bin path (for fairseq-generate)
    :param beam: beam size
    :param max_tokens: max tokens
    :param buffer_size: buffer size
    :return:
    """

    if ori_path is not None:
        generate = f"fairseq-interactive {data_path} --path {ckpt} --input {ori_path} " \
                   f"--beam {beam} --max-tokens {max_tokens} --buffer-size {buffer_size} > {system_out}"
        os.system(generate)

    elif gen_subset is not None:
        generate = f"fairseq-generate {data_path} --path {ckpt} --gen-subset {gen_subset} " \
                   f"--beam {beam} --max-tokens {max_tokens} --print-alignment > {system_out}"
        os.system(generate)
