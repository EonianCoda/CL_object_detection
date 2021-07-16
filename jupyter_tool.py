# VAL_ARG_SAMPLE = "--dataset voc2007 \
#                     --state 1 \
#                     --epoch 30 \
#                     --scenario 15 1 \
#                     --just_val True"
                    
# TRAIN_ARG_SAMPLE = "--dataset voc2007 \
#                     --start_epoch 1 \
#                     --end_epoch 100 \
#                     --start_state 0 \
#                     --end_state 0 \
#                     --scenario 10 10\
#                     --print_il_info True\
#                     --debug False \
#                     --record False"

def text_to_args(args):
    args = [arg.rstrip() for arg in args.split('--') if arg != '']
    result_arg = []
    for arg in args:
        texts = arg.split(' ')
        result_arg.append('--' + texts[0])
        for i in range(1, len(texts)):
            result_arg.append(texts[i])
    return result_arg