from oncopolicy.metrics.factory import RegisterMetric
import numpy as np
import pdb

MONTH_IN_STEP = 6

def abstract_mo_to_cancer(screen_progression, batch, args, scale_fn):
    '''
        args:
        - screen_progression: List of screen timesteps for each el in batch
        - batch: full info about batch
        - args: run timeargs
        - scale_fn: fn to scale reward given batch_details
    '''
    batch_size = batch['last_neg_time_step'].size()[0]
    last_neg_time_stamp_for_batch = batch['last_neg_time_step'].data.cpu().numpy()
    time_step_at_cancer_for_batch = batch['time_step_at_cancer'].data.cpu().numpy()
    will_get_cancer = batch['ever_has_cancer']

    cancer_month_accelerations = []
    for patient_indx, screens_points in enumerate(screen_progression):
        last_screen = max(screens_points)
        last_neg_step = last_neg_time_stamp_for_batch[patient_indx]
        cancer_step = time_step_at_cancer_for_batch[patient_indx]
        assert last_neg_step < cancer_step

        if not bool(will_get_cancer[patient_indx].item()) or last_screen < last_neg_step:
            cancer_month_accelerations.append(0)
            continue

        def in_early_detection_window(step):
            if args.use_pessimistic_detection_definition:
                max_early_steps = args.max_early_detection_benefit // MONTH_IN_STEP
                return step > last_neg_step and step >= (cancer_step - max_early_steps)
            else:
                return step > last_neg_step

        screens_after_last_neg = [step for step in screens_points if in_early_detection_window(step) ]
        if len(screens_after_last_neg)  == 0:
            cancer_month_accelerations.append(0)
            continue

        first_screen_after_last_neg = min(screens_after_last_neg)
        acceleration_in_mo = (cancer_step - first_screen_after_last_neg) *  MONTH_IN_STEP
        acceleration_in_mo = min(acceleration_in_mo, args.max_early_detection_benefit)

        month_delta = scale_fn( acceleration_in_mo, patient_indx, batch )
        cancer_month_accelerations.append(month_delta)

    assert len(cancer_month_accelerations) == batch_size
    return np.array(cancer_month_accelerations)

@RegisterMetric("mo_to_cancer")
def mo_to_cancer(screen_progression, batch, args):
    def scale_fn( acceleration_in_mo, patient_indx, batch ):
        return acceleration_in_mo
    return abstract_mo_to_cancer(screen_progression, batch, args, scale_fn)
