#
# python inspect_pb.py --frozen_graph=True --input=c:\Users\chungyeh\Downloads\frozen_vgg_16.pb --input_names=input --output_names=vgg_16/fc8/squeezed --output=optimized_frozen_vgg_16.pb
#

import argparse
import os
import sys

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import graph_util

FLAGS = None

def _node_name(n):
  if n.startswith("^"):
    return n[1:]
  else:
    return n.split(":")[0]

def _extract_graph_summary(graph_def):
  """Extracts useful information from the graph and returns them."""
  name_to_input_name = {}  # Keyed by the dest node name.
  name_to_node = {}  # Keyed by node name.

  # Keeps track of node sequences. It is important to still output the
  # operations in the original order.
  name_to_seq_num = {}  # Keyed by node name.
  seq = 0
  for node in graph_def.node:
    n = _node_name(node.name)
    ###if node.name == 'input':
    ###  print("input:", node)
    name_to_node[n] = node
    name_to_input_name[n] = [_node_name(x) for x in node.input]
    name_to_seq_num[n] = seq
    seq += 1
  return name_to_input_name, name_to_node, name_to_seq_num


def _assert_nodes_are_present(name_to_node, nodes):
  """Assert that nodes are present in the graph."""
  for d in nodes:
    assert d in name_to_node, "%s is not in graph" % d


def _bfs_for_reachable_nodes(target_nodes, name_to_input_name):
  """Breadth first search for reachable nodes from target nodes."""
  nodes_to_keep = set()
  # Breadth first search to find all the nodes that we should keep.
  next_to_visit = target_nodes[:]
  while next_to_visit:
    n = next_to_visit[0]
    del next_to_visit[0]
    if n in nodes_to_keep:
      # Already visited this node.
      continue
    nodes_to_keep.add(n)
    next_to_visit += name_to_input_name[n]
  return nodes_to_keep

def main(unused_args):
  if not gfile.Exists(FLAGS.input):
    print("Input graph file '" + FLAGS.input + "' does not exist!")
    return -1

  input_graph_def = graph_pb2.GraphDef()
  with gfile.Open(FLAGS.input, "rb") as f:
    data = f.read()
    if FLAGS.frozen_graph:
      input_graph_def.ParseFromString(data)
    else:
      text_format.Merge(data.decode("utf-8"), input_graph_def)

  ##output_graph_def = optimize_for_inference_lib.optimize_for_inference(
  ##    input_graph_def,
  ##    FLAGS.input_names.split(","),
  ##    FLAGS.output_names.split(","),
  ##    FLAGS.placeholder_type_enum,
  ##    FLAGS.toco_compatible)
  
  dest_nodes = FLAGS.output_names.split(",")
  
  name_to_input_name, name_to_node, name_to_seq_num = _extract_graph_summary(
      input_graph_def)
  ##_assert_nodes_are_present(name_to_node, dest_nodes)

  nodes_to_keep = _bfs_for_reachable_nodes(dest_nodes, name_to_input_name)

  nodes_to_keep_list = sorted(
      list(nodes_to_keep), key=lambda n: name_to_seq_num[n])
  
  # dump
  #for node_name in nodes_to_keep_list:
  #  print("name:", name_to_node[node_name].name)
  #  print("op:", name_to_node[node_name].op)
  #  print("input:", name_to_node[node_name].input)
  #return 0

  name_to_memory = {}
  name_to_param = {}
  name_to_shape = {}
  memory = 0
  params = 0
  for node_name in nodes_to_keep_list:
    if name_to_node[node_name].op in 'Placeholder':
      attr = name_to_node[node_name].attr['shape']
      shape = [x.size for x in attr.shape.dim]
      
      shape[0] = 1
      #shape[1] = 1024
      #shape[2] = 1024
      shape[3] = 3

      size = 1
      for s in shape:
        if int(s) < 0:
          size *= 1
        else:
          size *= int(s)
      #print("size:", size)

      name_to_memory[node_name] = size
      name_to_param[node_name] = 0
      name_to_shape[node_name] = shape
    elif name_to_node[node_name].op in 'Const':
      if 'paddings' in node_name:
        print(name_to_node[node_name])
        continue
        
      attr = name_to_node[node_name].attr['value']
      shape = [x.size for x in attr.tensor.tensor_shape.dim]

      size = 1
      for s in shape:
        if int(s) < 0:
          size *= 1
        else:
          size *= int(s)
      #print("size:", size)

      name_to_memory[node_name] = 0
      name_to_param[node_name] = size
      name_to_shape[node_name] = shape
    elif name_to_node[node_name].op in 'Identity':
        #print(name_to_node[node_name])
        continue
    elif name_to_node[node_name].op in 'BiasAdd':
        #print(name_to_node[node_name])
        for name in name_to_node[node_name].input:
          if "Conv2D" in name:
            shape = name_to_shape[name]
        name_to_memory[node_name] = 0
        name_to_param[node_name] = 0
        name_to_shape[node_name] = shape
        #print("BiasAdd[shape]:", shape)
        continue
    elif name_to_node[node_name].op in 'MaxPool':
        #if "pool5" in node_name:
        #  print(name_to_node[node_name])
        #continue
        #shape = name_to_node[node_name].attr['padding'].s
        for name in name_to_node[node_name].input:
          if "Relu" in name:
            shape = name_to_shape[name]
        shape[1] = shape[1] // 2 
        shape[2] = shape[2] // 2
        name_to_memory[node_name] = shape[1] * shape[2] * shape[3]
        name_to_param[node_name] = 0
        name_to_shape[node_name] = shape
    elif name_to_node[node_name].op in 'Relu':
        #print(name_to_node[node_name])
        for name in name_to_node[node_name].input:
          if "BiasAdd" in name:
            shape = name_to_shape[name]
        name_to_memory[node_name] = 0
        name_to_param[node_name] = 0
        name_to_shape[node_name] = shape
        #print("Relu[shape]:", shape)        
        continue
    elif name_to_node[node_name].op in 'Conv2D':
        #print(name_to_node[node_name])
        #continue
        #shape = name_to_node[node_name].attr['padding'].s
        for name in name_to_node[node_name].input:
          #print("Conv2D/input:", name)
          if "weights" in name:
            #attr = name_to_node[name[:-5]].attr['value']
            #shape = [x.size for x in attr.tensor.tensor_shape.dim]
            shape = name_to_shape[name[:-5]]
          elif 'dropout' in name:
            shape1 = name_to_shape[name_to_node[name].input[0]]
          else:
            shape1 = name_to_shape[name]
        
        #print("shape:{0}, shape1:{1}".format(shape, shape1))
        if "fc" in node_name:
          size = shape[3]
        else:    
          size = int(shape1[1])*int(shape1[2])*int(shape[3])
        #print("size:", size)
        shape[0] = shape1[3]
        shape[1] = shape1[1]
        shape[2] = shape1[2]
      
        name_to_memory[node_name] = size
        name_to_param[node_name] = 0
        name_to_shape[node_name] = shape
    elif name_to_node[node_name].op in 'Pad':
        print(name_to_node[node_name])
        name_to_memory[node_name] = 0
        name_to_param[node_name] = 0
        name_to_shape[node_name] = [1,224,224,3]
    else:
      name_to_shape[node_name] = "no shape"        
      name_to_memory[node_name] = 0
      name_to_param[node_name] = 0

    memory += name_to_memory[node_name] 
    params += name_to_param[node_name]
    print("name:{0} op:{1} shape:{2} memory:{3} param:{4}".format(node_name, name_to_node[node_name].op, name_to_shape[node_name],
        name_to_memory[node_name], name_to_param[node_name]))
    if name_to_node[node_name].op in 'MaxPool':
      print("")
      
  #print("Total memory:", memory, " total params:", params)
  print("Total memory:", int(memory*4/1024/1024), "MB total params:", int(params*4/1024/1024), "MB") 
                      
  return 0

def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--input",
      type=str,
      default="c:/Users/chungyeh/Downloads/frozen_vgg_16.pb",
      help="TensorFlow \'GraphDef\' file to load.")
  parser.add_argument(
      "--output",
      type=str,
      default="",
      help="File to save the output graph to.")
  parser.add_argument(
      "--input_names",
      type=str,
      default="input",
      help="Input node names, comma separated.")
  parser.add_argument(
      "--output_names",
      type=str,
      default="vgg_16/fc8/squeezed",
      help="Output node names, comma separated.")
  parser.add_argument(
      "--frozen_graph",
      nargs="?",
      const=True,
     type="bool",
      default=True,
      help="""\
      If true, the input graph is a binary frozen GraphDef
      file; if false, it is a text GraphDef proto file.\
      """)
  parser.add_argument(
      "--placeholder_type_enum",
      type=int,
      default=dtypes.float32.as_datatype_enum,
      help="The AttrValue enum to use for placeholders.")
  parser.add_argument(
      "--toco_compatible",
      type=bool,
      default=False,
      help="""\
      If true, only use ops compatible with Tensorflow
      Lite Optimizing Converter.\
      """)
  return parser.parse_known_args()


if __name__ == "__main__":
  FLAGS, unparsed = parse_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
