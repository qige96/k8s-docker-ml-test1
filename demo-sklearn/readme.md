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
```

## Step 2: Build a docker image and push to a registry

Write such a [file](https://docs.docker.com/engine/reference/builder/) called `Dockerfile` to specify how the image should be built

```dockerfile
# Build our image on top of a public data science images
# https://hub.docker.com/r/jupyter/scipy-notebook/
From jupyter/scipy-notebook:latest

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
docker build --tag demo-sklearn:0.1 .
```

Now we push the image to a registry. There a many hosting services, like docker hub, github, etc. Here I chose github:

```shell
# before pushing, we need to first login
docker login ghcr.io --username qige96 --password <Personal_Access_Token>
# Now we can push
docker tag demo-sklearn:0.1 ghcr.io/qige96/k8s-docker-ml-test1:demo-sklearn
docker push ghcr.io/qige96/k8s-docker-ml-test1:demo-sklearn
```


### Step 3: Pull the image and run the job

Now login to the EIDF cluster.

```shell
ssh -J <username>@eidf-gateway.epcc.ed.ac.uk <username>@<cluster-addr>
```

To launch a k8s job, we first write [k8s spec file](https://kubernetes.io/docs/concepts/overview/working-with-objects/kubernetes-objects/) called `demo-sklearn.yml`

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: demo-sklearn-k8s
spec:
  template:
    spec:
      containers:
      - name: tdemo-sklearn-k8s
        imagePullPolicy: Always
        image: ghcr.io/qige96/k8s-docker-ml-test1:demo-sklearn
        command: ["python3",  "train.py"]
      restartPolicy: Never
  backoffLimit: 0
````

Then we can launch the job by one line of command

```shell
kubectl apply -f demo-sklearn.yml --validate=false
```
After launching the k8s job, we can see a pod `demo-sklearn-k8s-mqmjh` created 
```shell
rzhu-infk8s@eidf029-host1:~$ kubectl get pods
NAME                          READY   STATUS              RESTARTS   AGE
demo-sklearn-k8s-mqmjh        0/1     ContainerCreating   0          6s
node-info-80gb-full-8-b6f69   0/1     Completed           0          119m
nvidia-ubuntu-aryo-1x80gpus   1/1     Running             0          29h
nvidia-ubuntu-aryo-2x80gpus   1/1     Running             0          33h
```
After a while, when we see the status of the of `demo-sklearn-k8s-mqmjh` becomes "Completed"
```shell
rzhu-infk8s@eidf029-host1:~$ kubectl get pods
NAME                          READY   STATUS      RESTARTS   AGE
demo-sklearn-k8s-mqmjh        0/1     Completed   0          2m14s
node-info-80gb-full-8-b6f69   0/1     Completed   0          121m
nvidia-ubuntu-aryo-1x80gpus   1/1     Running     0          29h
```

We can inspect the command line printing outputs of our sklearn job by displaying the logs of the pod

```shell
rzhu-infk8s@eidf029-host1:~$ kubectl logs demo-sklearn-k8s-mqmjh
accuracy_score: 1.0
Done!
```

Clean up
----------
Finally, delete the jobs and pods on the cluster (being a considerate user)

```shell
kubectl delete jobs demo-sklearn-k8s # the name of our k8s job
```
This command will delete the job and the related pods.

