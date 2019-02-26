#import rouge
import pyrouge
import logging

from config import CONFIG as conf
ref_folder = conf['ref_folder']
pred_folder = conf['pred_folder']

def rouge_eval(ref_str, pred_str):
    """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
    for i in range(len(ref_str)):
        with open(ref_folder+str(i)+'_reference.txt', 'w') as f_out:
            f_out.write(ref_str[i])
        with open(pred_folder+str(i)+'_decoded.txt', 'w') as f_out:
            f_out.write(pred_str[i])
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_folder
    r.system_dir = pred_folder
    logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
    rouge_results = r.convert_and_evaluate()
    return r.output_to_dict(rouge_results)

def prepare_results(metric, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


def compute_rouge_score(preds, refs):
    #for aggregator in ['Avg', 'Best', 'Individual']:
    ret_scores = []
    for aggregator in ['Avg']:
        print('Evaluation with {}'.format(aggregator))
        apply_avg = aggregator == 'Avg'
        apply_best = aggregator == 'Best'

        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                               max_n=2,
                               limit_length=True,
                               length_limit=2000,
                               length_limit_type='words',
                               apply_avg=apply_avg,
                               apply_best=apply_best,
                               alpha=0.5, # Default F1_score
                               weight_factor=1.2,
                               stemming=True)



        #all_hypothesis = [hypothesis_1, hypothesis_2]
        #all_references = [references_1, references_2]
        all_hypothesis = preds
        all_references = refs

        scores = evaluator.get_scores(all_hypothesis, all_references)

        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
                for hypothesis_id, results_per_ref in enumerate(results):
                    nb_references = len(results_per_ref['p'])
                    for reference_id in range(nb_references):
                        print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                        print('\t' + prepare_results(metric, results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
                print()
            else:
                print(prepare_results(metric, results['p'], results['r'], results['f']))
                ret_scores.append(results['f'])
        print()
    return ret_scores[:3]
