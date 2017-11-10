

def do_analyse_sst(file_path, dev=True, delta=0, stop=None):
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        find_entry = False
        output = [0, 0., 0., 0., 0., 0., 0.] # xx, dev, test,
        for line in file:
            if not find_entry:
                if line.startswith('data round'):  # get step
                    output[0] = int(line.split(' ')[-4].split(':')[-1])
                    if stop is not None and output[0] > stop: break
                if line.startswith('==> for dev'):  # dev
                    output[1] = float(line.split(' ')[-1])
                    output[2] = float(line.split(' ')[-4][:-1])
                    output[3] = float(line.split(' ')[-6][:-1])
                    find_entry = True
            else:
                if line.startswith('~~> for test'):  # test
                    output[4] = float(line.split(' ')[-1])
                    output[5] = float(line.split(' ')[-4][:-1])
                    output[6] = float(line.split(' ')[-6][:-1])
                    results.append(output)
                    find_entry = False
                    output = [0, 0., 0., 0., 0., 0., 0.]

    # max step
    if len(results) > 0:
        print('max step:', results[-1][0])

    # sort
    sort = 1 if dev else 4
    sort += delta

    output = list(sorted(results, key=lambda elem: elem[sort], reverse=(not delta == 2)))

    for elem in output[:20]:
        print('step: %d, dev_sent: %.4f, dev: %.4f, dev_loss: %.4f, '
              'test_sent: %.4f, test: %.4f, test_loss: %.4f' %
              (elem[0], elem[1], elem[2], elem[3],elem[4], elem[5],elem[6]))


if __name__ == '__main__':

    file_path = '/Users/tshen/Desktop/tmp/file_transfer/sst/Jul-22-17-56-41_log.txt'
    dev = True
    delta = 0

    do_analyse_sst(file_path, dev, delta, None)


