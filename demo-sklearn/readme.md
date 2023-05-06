Run a Scikit-Learn job in the k8s environment
=============================================

> Credits to this [blogpost](https://blog.dataiku.com/how-to-perform-basic-ml-training-with-scikit-learn-docker-and-kubernetes)

This document outlines the main steps to run a simple machine learning experiment in the kubernetes environment. We will train an SVM classifier on the iris dataset using the `scikit-learn` package, and show the prediction accuracy of the trained model.

Here is the main workflow
1. prepare all the project code and data on our laptop
2. build the project to be a docker image and push the image to a registry
3. lanuch a k8s job on the EIDF cluster to pull the image and run our ML experiment

Set up
-------
1. Have access to the EIDF cluster
2. Have docker installed on our own laptop
3. Have an account of an image registry (docker hub, github, etc.)


Minimal Experiment
--------------------

### Step 1: Prepare the python script

Create a project on our ownn laptop. Below is the main python script `sklearn_job.py` that runs our ML job

```python
import os
# to avoid the "OpenBLAS blas_thread_init: pthread_create failed for thread 1 of 8: Operation not permitted" error
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump

# Initialise an SVM classifier
clf = svm.SVC()

# Load and split data
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# train and predict
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# display results
print(f'accuracy_score: {accuracy_score(y_test, y_pred)}')
print("Done!")

# outputs
dump(clf, 'svc-iris.model')
with open('report.txt', 'w') as f:
    f.write(f'accuracy_score: {accuracy_score(y_test, y_pred)}\n')
```

## Step 2: Build a docker image and push to a registry

Write such a [file](https://docs.docker.com/engine/reference/builder/) called `Dockerfile` to specify how the image should be built

```dockerfile
# Build our image on top of a public data science images
# https://hub.docker.com/r/jupyter/scipy-notebook/
From jupyter/scipy-notebook:latest

# the docker runs as the root user
USER root

# Our image just include the main python script
COPY sklearn_job.py ./sklearn_job.py
```

Now on our own laptop, the project sturcture looks like this 

```shell
ricky@ricky-pc$ tree .
sklearn_job.py
Dockerfile
```

Then, run the following commands to build  the image

```shell
docker build --tag demo-sklearn .
```

Now we push the image to a registry. There a many hosting services, like docker hub, github, etc. Here I chose github:

```shell
# before pushing, we need to first login
docker login ghcr.io --username qige96 --password <Personal_Access_Token>
# Now we can push
docker tag demo-sklearn ghcr.io/qige96/k8s-docker-ml-test1:demo-sklearn
docker push ghcr.io/qige96/k8s-docker-ml-test1:demo-sklearn
```


### Step 3: Pull the image and run the job

Now login to the EIDF cluster.

```shell
ssh -J <username>@eidf-gateway.epcc.ed.ac.uk <username>@<cluster-addr>
```

The simplest way to pull and run our image is:
```shell
kubectl run demo-sklearn --image=ghcr.io/qige96/k8s-docker-ml-test1:demo-sklearn
```
The a pod called `demo-sklearn` will be created, running a container made by our image.

Next, we login to the container and run out job.

```shell
rzhu-infk8s@eidf029-host1:~$ kubectl exec -it demo-sklearn -- /bin/bash
root@demo-sklearn:~#
```
From the change of the CLI prompt we can see that now we login to the container. Then we can run our job

```shell
root@demo-sklearn:~# python sklearn_job.py 
accuracy_score: 1.0
Done!
root@demo-sklearn:~# ls
report.txt  sklearn_job.py  svc-iris.model
```

We can see now we successfully run our ML job, and dump the model file to the container file system. Finally, we are going to move the output files from the container to our cluster directory

```shell
root@demo-sklearn:~# exit  # disconnect from the container and go back to the cluster
exit
rzhu-infk8s@eidf029-host1:~$ kubectl cp demo-sklearn:svc-iris.model ./svc-iric.model
rzhu-infk8s@eidf029-host1:~$ kubectl cp demo-sklearn:report.txt ./svc-iris-report.txt
rzhu-infk8s@eidf029-host1:~$ ls
svc-iric.model  svc-iris-report.txt
```


Clean up
----------
At last, delete the pods on the cluster (being a considerate user)

```shell
kubectl delete pods demo-sklearn-k8s 
```
This command will delete the pods.

