==========================
Frequently Asked Questions
==========================

**Why iterating the optimization?**

When working with complex high-dimensionality datasets, one may be interested in data relationships at different resolutions. In a pet image recognition project, one may want to distinguish not only cats from dogs but also different breeds.
While several hierarchical clustering methods are available, they generally tend to ignore the fact that optimal parameters in a typical clustering analysis are dependent on the subset of data being considered, and work instead on a single set resolution. 
The optimal embedding for separating dog breeds may lay on a different from the one that allows separating distinct species, while features that may be relevant/irrelevant in distinguishing a cat from a dog may hold negligible/considerable information at the breeds level. 
For a proper hierarchal analysis, the choice of clustering parameters should be repeated at each iteration, accounting for the new range and shape of the data subsets.
:code:`raccoon` identifies the proper clustering parameters at each hierarchical level, by repeating the optimization independently for each identified cluster.  

**Do I really need to run raccoon on my dataset?**

Not necessarily. This library is designed to identify complex clustering relationships in high-dimensional datasets.
It is most suited for data in which hierarchical groups are expected, and are untreatable in their original space.
If you know your analysis may benefit from a thorough dimensionality reduction, this library may be useful to you.
For simpler and/or low-dimensionality datasets, a more efficient out-of-the-box clustering algorithm could more than enough.
You can still run a parameters search with :code:`raccoon` even in these cases, but you need to be aware of the 
accuracy-computational cost trade-off. Know your data!

**Can I use my own clustering score?**

Of course. :code:`raccoon` can takes custom objective functions as long as they comply to the format used 
by other common internal validation scores available in sickit-learn, like the silhouette score. 
The score needs to be at a maximum when the clustering is optimal, invert its direction if necessary.
Just remember to make sure your function is compatible with RAPIDS if you are planning to use a GPU.

**Why is there an option to run tSVD before UMAP? I thought UMAP didn't need preprocessing.**

That is correct, in principle UMAP does not need a preliminary step of low-information removal,
but it may be the case that you want to remove useless and redundant features nevertheless, e.g. 
to help with the interpretation of the results, or to improve the signal-to-noise ratio in particularly complex dataset.
Our goal is a highly costumizable library, so the option of skip these steps is available.

**Can I still skip the low-information removal/non-linear dimensionality reduction?**

Yes, flags are available to skip the different steps of the analysis, however, this library was designed
with all of these steps in mind and you this choice may lead to a slight efficiency loss. 
See the documentation for more details.

**Can I use something else rather than UMAP?**

For the moment, the only alternative is to only run the low-information removal (e.g. with tSVD) and
skip the non-linear dimensionality reduction step, if you wish so. We plan on adding more algorithms in the future. Stay tuned!
