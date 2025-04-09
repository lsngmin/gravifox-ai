import tensorflow as tf
import tensorboard

log_dir = "/Users/sngmin/PycharmProjects/tvb-ai/logs"
tb = tensorboard.program.TensorBoard()

try:
    tb.configure(argv=[None, "--logdir", log_dir])
    tb.launch()
except Exception as e:
    print(f"Error occurred: {e}")
