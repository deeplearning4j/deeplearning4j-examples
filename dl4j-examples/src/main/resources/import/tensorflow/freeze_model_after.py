import tensorflow as tf
from tensorflow.python.tools import freeze_graph

# change this to where your model was saved in the other python file
save_dir = "/Users/susaneraly/Desktop/saved_model"

if "susaneraly" in save_dir:
    raise ValueError("Please change save_dir to the local path your original model was saved to")

try:
    checkpoint = tf.train.get_checkpoint_state(save_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
except:
    raise ValueError("Could not read checkpoint state for path {}".format("/Users/susaneraly/Desktop/saved_model"))

input_graph = "{}/model.txt".format(save_dir)
output_graph = "{}/frozen_model.pb".format(save_dir)
# You can also run this on the command line which is much easier
# More information on freezing graphs are found in the TensorFlow documentation
#       https://www.tensorflow.org/extend/tool_developers/#freezing
# Modify these in accordance with your graph! These are the default options..TF complains if these are not provided. Hence cleaner to run on the command line
freeze_graph.freeze_graph(input_graph=input_graph,
                          input_saver="",
                          input_checkpoint=input_checkpoint,
                          output_graph=output_graph,
                          input_binary=False,
                          output_node_names="output",  # change this to match your graph
                          restore_op_name="save/restore_all",
                          filename_tensor_name="save/Const:0",
                          clear_devices=True,
                          initializer_nodes="")

with tf.gfile.GFile(output_graph, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.train.write_graph(graph_def, save_dir, 'frozen_graph.pbtxt', True)

