from roboflow import Roboflow
rf = Roboflow(api_key="PpnB151yMH8KXwJ43TKh")
project = rf.workspace("dentex").project("dentex2")
dataset = project.version(2).download("coco")
