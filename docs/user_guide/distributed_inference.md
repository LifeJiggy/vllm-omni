# Distributed Inference Orchestration

vLLM-Omni supports distributed inference across multiple GPUs and nodes for high-throughput, fault-tolerant multimodal generation.

## Overview

The distributed inference system coordinates multiple Omni instances across different nodes, providing:

- **Load Balancing**: Automatic distribution of requests across available nodes
- **Fault Tolerance**: Automatic failure detection and recovery
- **Scalability**: Easy scaling from single-node to multi-node deployments
- **High Availability**: Continues operation even when individual nodes fail

## Architecture

```
┌─────────────────┐
│   API Requests  │
└─────────────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐
│ Load Balancer   │───▶│ Health Monitor  │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│   Node 1        │    │   Node 2        │
│ (Omni Instance) │    │ (Omni Instance) │
└─────────────────┘    └─────────────────┘
```

## Quick Start

### Basic Usage

```python
from vllm_omni.distributed import DistributedOrchestrator

# Initialize distributed orchestrator
orchestrator = DistributedOrchestrator(
    model="Qwen/Qwen2.5-Omni-7B",
    num_nodes=4,
    ray_address="auto"  # Ray cluster address
)

# Generate across distributed nodes
results = await orchestrator.generate([
    "Generate an image of a sunset",
    "Create a picture of a mountain",
    "Draw a forest landscape"
])

for result in results:
    print(f"Generated: {result}")
```

### Advanced Configuration

```python
# Custom load balancing strategy
orchestrator = DistributedOrchestrator(
    model="Qwen/Qwen2.5-Omni-7B",
    num_nodes=8,
    load_balancing_strategy="least_loaded",  # or "round_robin"
    health_check_interval=10.0,  # seconds
    max_retries=3
)
```

## Load Balancing Strategies

### Round Robin
Distributes requests evenly across all healthy nodes in circular order.

```python
orchestrator = DistributedOrchestrator(
    model="model",
    num_nodes=4,
    load_balancing_strategy="round_robin"
)
```

### Least Loaded
Routes requests to the node with the lowest current load.

```python
orchestrator = DistributedOrchestrator(
    model="model",
    num_nodes=4,
    load_balancing_strategy="least_loaded"
)
```

## Fault Tolerance

### Automatic Failure Detection

The system continuously monitors node health:

```python
# Check cluster health
stats = orchestrator.get_stats()
print(f"Healthy nodes: {stats['healthy_nodes']}")
print(f"Unhealthy nodes: {stats['unhealthy_nodes']}")
```

### Failure Recovery

When a node fails, the system automatically:

1. **Detects** the failure through health checks
2. **Marks** the node as unhealthy
3. **Redistributes** pending requests to healthy nodes
4. **Retries** failed requests with exponential backoff

```python
# Generate with automatic retry on failures
results = await orchestrator.generate(
    prompts=["prompt1", "prompt2"],
    max_retries=3  # Retry failed requests up to 3 times
)
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_nodes` | 2 | Number of worker nodes |
| `load_balancing_strategy` | "round_robin" | Load distribution strategy |
| `health_check_interval` | 5.0 | Health check frequency (seconds) |
| `max_retries` | 2 | Maximum retry attempts for failed requests |
| `ray_address` | None | Ray cluster address |

## Monitoring and Metrics

### Cluster Statistics

```python
stats = orchestrator.get_stats()
print(f"""
Cluster Status:
- Total nodes: {stats['num_nodes']}
- Healthy nodes: {len(stats['healthy_nodes'])}
- Load distribution: {stats['load_distribution']}
- Health details: {stats['health_stats']}
""")
```

### Health Monitoring

```python
from vllm_omni.distributed import HealthMonitor

monitor = HealthMonitor.remote()
health_stats = await monitor.get_health_stats.remote()
```

## Deployment Examples

### Single Machine, Multi-GPU

```python
# 4 GPUs on single machine
orchestrator = DistributedOrchestrator(
    model="Qwen/Qwen2.5-Omni-7B",
    num_nodes=4,
    ray_address=None  # Local Ray cluster
)
```

### Multi-Node Cluster

```python
# Ray cluster across multiple machines
orchestrator = DistributedOrchestrator(
    model="Qwen/Qwen2.5-Omni-7B",
    num_nodes=16,
    ray_address="ray://head-node:10001"
)
```

### Kubernetes Deployment

```yaml
apiVersion: ray.io/v1alpha1
kind: RayCluster
metadata:
  name: vllm-omni-cluster
spec:
  rayVersion: '2.9.0'
  headGroupSpec:
    replicas: 1
    rayStartParams:
      block: 'true'
    template:
      spec:
        containers:
        - name: ray-head
          image: vllm-omni:latest
  workerGroupSpecs:
  - replicas: 4
    minReplicas: 1
    maxReplicas: 8
    groupName: worker-group
    rayStartParams:
      block: 'true'
    template:
      spec:
        containers:
        - name: ray-worker
          image: vllm-omni:latest
          resources:
            limits:
              nvidia.com/gpu: 1
```

## Performance Considerations

### Optimal Node Count

- **Small workloads**: 2-4 nodes
- **Medium workloads**: 4-8 nodes
- **Large workloads**: 8-16+ nodes

### Memory Management

Ensure adequate memory per node:

```python
# Reserve memory for orchestration overhead
orchestrator = DistributedOrchestrator(
    model="model",
    num_nodes=4,
    memory_buffer_gb=2  # Reserve 2GB per node for overhead
)
```

### Network Bandwidth

For multi-node deployments, ensure sufficient network bandwidth between nodes for model synchronization and result transmission.

## Troubleshooting

### Common Issues

**No healthy nodes available**
- Check Ray cluster status: `ray status`
- Verify node resources and GPU availability
- Check network connectivity between nodes

**Slow performance**
- Monitor load distribution with `get_stats()`
- Consider switching load balancing strategies
- Check for network bottlenecks

**Frequent failures**
- Reduce `health_check_interval` for more frequent monitoring
- Increase `max_retries` for better fault tolerance
- Check system logs for underlying issues

## API Reference

### DistributedOrchestrator

Main entry point for distributed inference.

**Methods:**
- `generate(prompts, **kwargs)`: Generate outputs across distributed nodes
- `get_stats()`: Get cluster statistics
- `close()`: Clean up resources

### LoadBalancer

Handles request distribution across nodes.

**Methods:**
- `distribute_requests(prompts, healthy_nodes)`: Distribute requests
- `get_load_stats()`: Get current load information

### HealthMonitor

Monitors node health and manages failures.

**Methods:**
- `get_healthy_nodes()`: Get list of healthy nodes
- `get_health_stats()`: Get detailed health information