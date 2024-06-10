def jaccard(result):
    metrics = {'accuracy':0}
    N = 0
    for key in result.keys():
        metrics[key] = {'TP':0,'FP':0,'FN':0,'J':0,'N':0}
        metrics[key]['TP'] = result[key][key]
        for key2 in result.keys():
            N += result[key][key2]
            metrics[key]['N'] += result[key][key2]
            if key2 != key:
                metrics[key]['FP'] += result[key2][key]
                metrics[key]['FN'] += result[key][key2]
        
        metrics[key]['J'] = metrics[key]['TP']/(metrics[key]['TP']+metrics[key]['FP']+metrics[key]['FN'])
    
        metrics['accuracy'] += result[key][key]
    metrics['accuracy'] /= N

    return metrics
     