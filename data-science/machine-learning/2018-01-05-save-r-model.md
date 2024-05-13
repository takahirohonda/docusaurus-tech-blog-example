---
slug: data-science/machine-learning/how-to-save-machine-learning-models-in-r
title: How To Save Machine Learning Models In R
tags: [Data Science, Machine Learning, Save Models, R]
---

Once you finish training the model and are happy with it, you may need to consider saving the model. <!--truncate-->Otherwise, you will loose the model once you close the session. The model you create in R session is not persistent, only existing in the memory temporarily. Most of the time, training is a time-consuming process. Without saving the model, you have to run the training algorithm again and again. This is especially not good to happen in production.

In production, it is ideal to have a trained model saved and your code are only loading and using it to predict the outcome on the new dataset.

There are two ways to save and load models in R. Let’s have a look at them.

### save() and load()

You can save named object to a file and load it in the later session. With save(), you have to load it with the same name. Here is the example of saving the optimised neural networks model created in the previous post. The model name is model_nnet.

Let’s save the model. If you want to deploy it, you can push .rda file with your code to production.

```R
save(model_nnet, file = "/tmp/model_nnet.rda")
```

Once you successfully save it, close the current R session. Then, you can load it back in the new session. It’s ready for use.

```R
load(file = "/tmp/model_nnet.rda")
```

### saveRDS() and readRDS()

saveRDS() does not save the object name. Therefore, when you load it back, you can name it whatever you want (see below). While you can save many objects into a file with save(), saveRDS() only saves one object at a time as it is a lower-level function. saveRDS() also serialise an R object, which some people say better. But, most of the time, it really doesn’t matter. The model saved with save() and saveRDS() is the same. You will get the same predictive outcome.

```R
saveRDS(model_nnet, file = "/tmp/model_nnet2.rda")
```

As mentioned above, you can load it with whatever name you want (in this case, loaded as model2).

```R
model2 <- readRDS("/tmp/model_nnet2.rda")
```

Check to see if both models are loaded.

```
> ls()
> [1] "model_nnet" "model2"
> For sanity check, you can plot them.

plotnet(model_nnet)
plotnet(model2)
```

Common Warning Handling

When you try to save the model with either save() or saveRDS(), you may get the warnings such as ‘display list redraw incomplete’ and ‘invalid graphics state’ as below.

Warning messages:

```
1: In doTryCatch(return(expr), name, parentenv, handler) :
display list redraw incomplete
2: In doTryCatch(return(expr), name, parentenv, handler) :
invalid graphics state
3: In doTryCatch(return(expr), name, parentenv, handler) :
invalid graphics state
```

When you have these warnings, you need to reset your graphic device by running dev.off(). For example, if you have a plot in the R Studio viewer, you will probably get them. Once you reset it with dev.off(), you can save the model without warning.

Yep, now you are an expert in saving R models!
