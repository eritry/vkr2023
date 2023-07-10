import utils
import metrics


# заставляет x264 выдать точно нужный битрейт, сам он не справляется например в случае ippppppp
def get_exact_bitrate(inp, enc, dec, res, fr_n, bitr):
    bitrate_target = bitr
    bitrate_x = 0
    for p in [100, 10, 3]:
        while (bitrate_x < bitr):
            bitrate_target += p
            out, err = utils.encode(inp, enc, res, fr_n, bitrate=round(bitrate_target), keyint=fr_n)
            _, _ = utils.decode(enc, dec)
            psnr_x, _ = metrics.calculate_psnr(inp, dec, res, fr_n)
            bitrate_x = utils.calculate_bitrate(enc, fr_n)
        while (bitrate_x > bitr):
            bitrate_target -= p
            out, err = utils.encode(inp, enc, res, fr_n, bitrate=round(bitrate_target), keyint=fr_n)
            _, _ = utils.decode(enc, dec)
            psnr_x, _ = metrics.calculate_psnr(inp, dec, res, fr_n)
            bitrate_x = utils.calculate_bitrate(enc, fr_n)            
    
    out, err = utils.encode(inp, enc, res, fr_n, bitrate=round(bitrate_target), keyint=fr_n)
    _, _ = utils.decode(enc, dec)
    psnr_x264, _ = metrics.calculate_psnr(inp, dec, res, fr_n)
    bitrate_x264 = utils.calculate_bitrate(enc, fr_n)
    return psnr_x264, bitrate_x264

# считывает файл статистики и возвращает результат генетики
def get_ga_results(inp, enc, dec, res, fr_n, stats, bt):
    qps = []
    qp = []
    typ = []
    lines = []
    
    for line in open(stats): lines.append(line)
    
    i = 0
    j = 0
    while i < len(lines):
        if "QP" in lines[i]: 
            qp = lines[i + 1].split() # значения qp лучшего индивида данной эпохи
        if "TP" in lines[i]:
            typ = lines[i + 1].split() # значения типов кадров этого индивида, при режиме без типов кадров это нужно убрать
        i += 2
        j += 1

    # кладем только последний "CHECKED QP", он должен быть самым лучшим проверенным (с посчитанным результатом
    # при запуске кодера и декодера), также в предыдущем цикле можно положить других индивидов для проверки,
    # возможно кто-то окажется лучше, но наверняка мы не знаем, так как там лучший индивид поколения выбирался
    # по оценке фитнесс функции, без запуска кодера и декодера
    qps.append((qp, typ))
         
    max_psnr = 0
    bitr = 0
    qp_path = '../qp_st.txt'

    # выбор лучшего индивида из qps при соблюдении ограничения на битрейт
    for qp, typ in list(qps):
        qp = list(qp)
        typ = list(typ)
  
        with open(qp_path, 'w') as f:
            for i in range(len(qp)): 
                c = 'I' if typ[i] == '0' else 'P'
                print(i, c, qp[i], file=f)

        out, err = utils.encode(inp, enc, res, fr_n, qpfile=qp_path, preset='veryslow')
        out, err = utils.decode(enc, dec)
        psnr, _ = metrics.calculate_psnr(inp, dec, res, fr_n)
        bitrate = utils.calculate_bitrate(enc, fr_n)
        if bitrate <= bt and psnr > max_psnr:
            max_psnr = psnr
            bitr = bitrate
            
    return max_psnr, bitr


enc = '../enc_st.yuv'
dec = '../dec_st.yuv'
res = (352, 288)
qp_path = '../qp_st.txt'
files = [
    ('container', 100),
    ('akiyo', 30)
]

with open('res.txt', 'w') as outfile:
    frames = utils.read_frames()
    for kk in range(len(files)):
        name, bt = files[kk]
        inp = '../dataset/' + name + '.yuv'
        fr_n = frames[name]
        stats = '../stats/ipp/'+ name + '_' + str(bt) + '_60_' + str(fr_n) + '.best'

        max_psnr, bitr = get_ga_results(inp, enc, dec, res, fr_n, stats, bt)

        print(name, bt, '\tga\t psnr:', max_psnr, 'bitrate:', bitr)
        print(name, bt, '\tga\t psnr:', max_psnr, 'bitrate:', bitr, file=outfile)

        psnr_x264, bitrate_x264 = get_exact_bitrate(inp, enc, dec, res, fr_n, bt)

        print(name, bt, '\tx264\t psnr:', psnr_x264, 'bitrate:', bitrate_x264)
        print(name, bt, '\tx264\t psnr:', psnr_x264, 'bitrate:', bitrate_x264, file=outfile)
