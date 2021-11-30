# 1.-Fish-Classification-Project-for-Fish-Ecologist

A link to Comprehensive report on the dataset: https://groups.inf.ed.ac.uk/f4k/GROUNDTRUTH/RECOG/

Fish Recognition Ground-Truth data

This fish data is acquired from a live video dataset resulting in 27370 verified fish images. 
The whole dataset is divided into 23 clusters and each cluster is presented by a representative species, 
which is based on the synapomorphies characteristic from the extent that the taxon is monophyletic. 
The representative image indicates the distinction between clusters shown in the figure below, 
e.g. the presence or absence of components (anal-fin, nasal, infraorbitals), 
specific number (six dorsal-fin spines, two spiny dorsal-fins), particular shape (second dorsal-fin spine long), etc. 
This figure shows the representative fish species name and the numbers of detections. 
The data is very imbalanced where the most frequent species is about 1000 times more than the least one. 
The fish detection and tracking software described in [1] is used to obtain the fish images. 
The fish species are manually labeled by following instructions from marine biologists [2].

This data is organized into 23 groups, where the fish images and their masks are stored separately. 
Each cluster has a single package. The image files are named as "tracking id_fish id". Fish images with the same "tracking id" means they are belong to the same trajectory. 
"fish id" is a global unique id, which ranges from 1 to 27370. A reverse table contains "file name verse cluster id" is provided at here. 
The whole package of all groups is available here (510,912,000 bytes, checkSum).