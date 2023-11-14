# Spark deployment on NDCs

## Prerequisites

- A cluster of 3 machines, each with Docker installed.
- Network connectivity between all machines.
- Basic knowledge of Docker and Apache Spark.

## Step 1: Pull Docker Image for Apache Spark

You can use an official or custom Docker image for Apache Spark. For this tutorial, we will use the `bitnami/spark` image, which is well maintained and easy to use.

On each machine, run:

```shell
docker pull bitnami/spark
```

This will pull the latest Spark Docker image from the Bitnami repository.

## Step 2: Set Up the Master Node

Choose one of the machines to be the master node. On this machine, start the Spark master container with the following command:

```shell
docker run -d --name spark-master \
  -p 7077:7077 -p 8080:8080 \
  bitnami/spark:latest \
  /opt/bitnami/spark/bin/spark-class org.apache.spark.deploy.master.Master
```

Here's what this command does:

- `-d` runs the container in detached mode.
- `--name spark-master` names the container for easier reference.
- `-p 7077:7077` maps the Spark master service port.
- `-p 8080:8080` maps the Spark master web UI port.
- The Spark master service is started with the `spark-class` command.

## Step 3: Set Up Worker Nodes

On the other two machines, you will start the worker nodes. You need to link these to the master node.

```shell
docker run -d --name spark-worker \
  -p 8081:8081 \
  --link <master-node-ip>:master \
  bitnami/spark:latest \
  /opt/bitnami/spark/bin/spark-class org.apache.spark.deploy.worker.Worker spark://master:7077
```

Make sure to replace `spark-master` with the actual IP address or hostname of the master node if they are not on the same Docker network.

This command:

- Names the container `spark-worker`.
- Maps the worker web UI port.
- Links to the master container.
- Starts the Spark worker and connects it to the master node.

## Step 4: Verify the Cluster

After setting up the master and worker nodes, you should verify that the workers are correctly registered with the master.

1. Open a web browser and navigate to `http://<master-node-ip>:8080`, where `<master-node-ip>` is the IP address or hostname of the master node.
2. You should see the Spark web UI with information about the master and the connected workers.

## Step 5: Running a Spark Job

With the cluster set up, you can now submit Spark jobs to the master node.

Here is an example of how to run a Spark job:

```shell
docker exec spark-master \
  /opt/bitnami/spark/bin/spark-submit \
  --master spark://master:7077 \
  --class org.apache.spark.examples.SparkPi \
  /opt/bitnami/spark/examples/jars/spark-examples_2.12-3.0.0.jar 1000
```

This command:

- Executes the `spark-submit` command inside the `spark-master` container.
- Specifies the master URL.
- Runs the `SparkPi` example class.
- Uses the example jar included with the Spark image.

## Notes

- Ensure Docker’s networking allows the containers to communicate with each other across the machines.
- It's possible to create a custom Docker network for better control over the networking.
- Adjust the ports if you have a conflict with existing services.
- For a production setup, consider using Docker Swarm or Kubernetes for orchestration.


TODO: document this:

```bash
$SPARK_HOME/bin/spark-submit --master spark://10.11.0.61:7077  --executor-memory 16G --driver-memory 2G --num-executors 2 --executor-cores 10 llama2_test.py
```

On master :

```bash
docker run -d --name spark-master \
  -p 7077:7077 -p 8080:8080 \
  bitnami/spark:latest \
  /opt/bitnami/spark/bin/spark-class org.apache.spark.deploy.master.Master
```


On workers : 

```bash
docker run -d --name spark-worker   -p 8081:8081 bitnami/spark:latest   /opt/bitnami/spark/bin/spark-class org.apache.spark.deploy.worker.Worker spark://10.11.0.201:7077
```