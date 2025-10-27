"""
Architecture Diagram Generator

This script generates visual representations of the system architecture
using Graphviz. It creates multiple diagrams showing:
- System architecture
- Data pipeline
- Model architectures
- Deployment architecture
"""

import graphviz
from pathlib import Path


def generate_system_architecture():
    """Generate system architecture diagram."""
    dot = graphviz.Digraph(
        'system_architecture',
        comment='IoT Botnet Detection System Architecture'
    )
    dot.attr(rankdir='TB', splines='ortho', fontname='Arial')

    # Define styles
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')

    # Main components
    with dot.subgraph(name='cluster_data') as c:
        c.attr(label='Data Layer', style='filled', fillcolor='lightgray')
        c.node('dataset', 'N-BaIoT Dataset\n(9 IoT Devices)')
        c.node('download', 'Data Download\nScript')
        c.node('storage', 'Data Storage\n(CSV Files)')

    with dot.subgraph(name='cluster_preprocessing') as c:
        c.attr(label='Preprocessing Layer', style='filled', fillcolor='lightgray')
        c.node('loader', 'Data Loader')
        c.node('feature_eng', 'Feature Engineering\n(115 Features)')
        c.node('fisher', 'Fisher Score\nFeature Selection')
        c.node('normalization', 'StandardScaler\nNormalization')

    with dot.subgraph(name='cluster_models') as c:
        c.attr(label='Model Layer', style='filled', fillcolor='lightgray')
        c.node('anomaly', 'Anomaly Detection\n(Autoencoder)', fillcolor='lightgreen')
        c.node('classifier', 'Classification\n(Neural Network)', fillcolor='lightgreen')
        c.node('federated', 'Federated Learning\n(Flower)', fillcolor='lightyellow')

    with dot.subgraph(name='cluster_evaluation') as c:
        c.attr(label='Evaluation Layer', style='filled', fillcolor='lightgray')
        c.node('metrics', 'Performance Metrics\n(Accuracy, Precision, Recall)')
        c.node('lime', 'LIME Explainability')
        c.node('analysis', 'Overfitting Analysis')

    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Output Layer', style='filled', fillcolor='lightgray')
        c.node('models_out', 'Trained Models\n(.keras)')
        c.node('logs', 'Training Logs\n(TensorBoard)')
        c.node('reports', 'Analysis Reports')

    # Connections
    dot.edge('dataset', 'download')
    dot.edge('download', 'storage')
    dot.edge('storage', 'loader')
    dot.edge('loader', 'feature_eng')
    dot.edge('feature_eng', 'fisher')
    dot.edge('fisher', 'normalization')

    dot.edge('normalization', 'anomaly')
    dot.edge('normalization', 'classifier')
    dot.edge('normalization', 'federated')

    dot.edge('anomaly', 'metrics')
    dot.edge('classifier', 'metrics')
    dot.edge('federated', 'metrics')

    dot.edge('classifier', 'lime')
    dot.edge('metrics', 'analysis')

    dot.edge('anomaly', 'models_out')
    dot.edge('classifier', 'models_out')
    dot.edge('federated', 'models_out')

    dot.edge('anomaly', 'logs')
    dot.edge('classifier', 'logs')
    dot.edge('federated', 'logs')

    dot.edge('analysis', 'reports')
    dot.edge('lime', 'reports')

    return dot


def generate_data_pipeline():
    """Generate data pipeline diagram."""
    dot = graphviz.Digraph(
        'data_pipeline',
        comment='Data Processing Pipeline'
    )
    dot.attr(rankdir='LR', fontname='Arial')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')

    # Pipeline stages
    dot.node('raw', 'Raw Data\n(N-BaIoT CSV)', fillcolor='lightyellow')
    dot.node('load', 'Load CSV\npandas.read_csv()')
    dot.node('combine', 'Combine Devices\npd.concat()')
    dot.node('label', 'Add Labels\n(Benign/Mirai/Gafgyt)')
    dot.node('fisher_select', 'Feature Selection\nTop N by Fisher Score')
    dot.node('split', 'Train/Val/Test Split\n(70/15/15)')
    dot.node('scale', 'Standardization\nStandardScaler.fit(X_train)')
    dot.node('transform', 'Transform All Sets\nscaler.transform()')
    dot.node('ready', 'Training Ready\nNumPy Arrays', fillcolor='lightgreen')

    # Flow
    dot.edge('raw', 'load')
    dot.edge('load', 'combine')
    dot.edge('combine', 'label')
    dot.edge('label', 'fisher_select')
    dot.edge('fisher_select', 'split')
    dot.edge('split', 'scale', label='X_train')
    dot.edge('scale', 'transform')
    dot.edge('transform', 'ready')

    # Annotations
    dot.node('fisher_file', 'fisher.csv\n(Precomputed)',
             shape='note', fillcolor='lightyellow')
    dot.edge('fisher_file', 'fisher_select', style='dashed')

    return dot


def generate_autoencoder_architecture():
    """Generate autoencoder model architecture diagram."""
    dot = graphviz.Digraph(
        'autoencoder_architecture',
        comment='Autoencoder Model Architecture'
    )
    dot.attr(rankdir='TB', fontname='Arial')
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue')

    # Encoder
    with dot.subgraph(name='cluster_encoder') as c:
        c.attr(label='Encoder', style='filled', fillcolor='lightgray')
        c.node('input', 'Input Layer\n(115 features)', fillcolor='lightgreen')
        c.node('enc1', 'Dense Layer\n(100 neurons, ReLU)')
        c.node('enc2', 'Dense Layer\n(90 neurons, ReLU)')
        c.node('bottleneck', 'Bottleneck\n(90 neurons)', fillcolor='yellow')

    # Decoder
    with dot.subgraph(name='cluster_decoder') as c:
        c.attr(label='Decoder', style='filled', fillcolor='lightgray')
        c.node('dec1', 'Dense Layer\n(100 neurons, ReLU)')
        c.node('output', 'Output Layer\n(115 features)', fillcolor='lightcoral')

    # Flow
    dot.edge('input', 'enc1')
    dot.edge('enc1', 'enc2')
    dot.edge('enc2', 'bottleneck')
    dot.edge('bottleneck', 'dec1')
    dot.edge('dec1', 'output')

    # Loss
    dot.node('mse', 'Mean Squared Error\n(Reconstruction Loss)',
             shape='ellipse', fillcolor='lightyellow')
    dot.edge('output', 'mse')
    dot.edge('input', 'mse', style='dashed', label='target')

    return dot


def generate_classifier_architecture():
    """Generate classifier model architecture diagram."""
    dot = graphviz.Digraph(
        'classifier_architecture',
        comment='Classifier Model Architecture'
    )
    dot.attr(rankdir='TB', fontname='Arial')
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue')

    # Model layers
    dot.node('input', 'Input Layer\n(3-115 features)', fillcolor='lightgreen')
    dot.node('dense1', 'Dense Layer\n(64 neurons, ReLU)')
    dot.node('dropout1', 'Dropout\n(0.3)', fillcolor='lightyellow')
    dot.node('dense2', 'Dense Layer\n(32 neurons, ReLU)')
    dot.node('dropout2', 'Dropout\n(0.3)', fillcolor='lightyellow')
    dot.node('output', 'Output Layer\n(3 classes, Softmax)', fillcolor='lightcoral')

    # Flow
    dot.edge('input', 'dense1')
    dot.edge('dense1', 'dropout1')
    dot.edge('dropout1', 'dense2')
    dot.edge('dense2', 'dropout2')
    dot.edge('dropout2', 'output')

    # Classes
    dot.node('classes', 'Classes:\n- Benign\n- Mirai\n- Gafgyt',
             shape='note', fillcolor='lightyellow')
    dot.edge('output', 'classes', style='dashed')

    # Loss
    dot.node('loss', 'Categorical\nCrossentropy Loss',
             shape='ellipse', fillcolor='lightyellow')
    dot.edge('output', 'loss')

    return dot


def generate_federated_architecture():
    """Generate federated learning architecture diagram."""
    dot = graphviz.Digraph(
        'federated_architecture',
        comment='Federated Learning Architecture (Flower)'
    )
    dot.attr(rankdir='TB', fontname='Arial', splines='ortho')
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue')

    # Server
    dot.node('server', 'Flower Server\n(Aggregation)',
             fillcolor='lightcoral', shape='box3d')
    dot.node('strategy', 'FedAvg Strategy\n(Weighted Average)',
             fillcolor='lightyellow')

    # Clients
    with dot.subgraph(name='cluster_clients') as c:
        c.attr(label='IoT Device Clients (9)', style='filled', fillcolor='lightgray')
        c.node('client1', 'Danmini\nDoorbell', fillcolor='lightgreen')
        c.node('client2', 'Ecobee\nThermostat', fillcolor='lightgreen')
        c.node('client3', 'Philips B120N10\nBaby Monitor', fillcolor='lightgreen')
        c.node('client4', 'Provision PT-737E\nSecurity Camera', fillcolor='lightgreen')
        c.node('client5', 'Provision PT-838\nSecurity Camera', fillcolor='lightgreen')
        c.node('client6', 'Samsung SNH\nWebcam', fillcolor='lightgreen')
        c.node('client7', 'SimpleHome XCS7-1002\nSecurity Camera', fillcolor='lightgreen')
        c.node('client8', 'SimpleHome XCS7-1003\nSecurity Camera', fillcolor='lightgreen')
        c.node('client9', 'Ennio\nDoorbell', fillcolor='lightgreen')

    # Global model
    dot.node('global_model', 'Global Model\n(Aggregated Weights)',
             fillcolor='yellow', shape='cylinder')

    # Connections
    dot.edge('server', 'strategy', style='dashed')
    dot.edge('strategy', 'global_model', label='aggregate')

    # Server to clients (broadcast)
    for i in range(1, 10):
        dot.edge('global_model', f'client{i}',
                label='broadcast' if i == 1 else '',
                color='blue', style='dashed')

    # Clients to server (updates)
    for i in range(1, 10):
        dot.edge(f'client{i}', 'strategy',
                label='updates' if i == 1 else '',
                color='red')

    # Local data
    with dot.subgraph(name='cluster_data') as c:
        c.attr(label='Local Device Data', style='filled', fillcolor='lightyellow')
        c.node('data1', 'Device 1\nTraffic Data')
        c.node('data2', 'Device 2\nTraffic Data')
        c.node('data_etc', '...')

    dot.edge('data1', 'client1', style='dotted')
    dot.edge('data2', 'client2', style='dotted')
    dot.edge('data_etc', 'client9', style='dotted')

    return dot


def generate_deployment_architecture():
    """Generate deployment architecture diagram (future vision)."""
    dot = graphviz.Digraph(
        'deployment_architecture',
        comment='Production Deployment Architecture'
    )
    dot.attr(rankdir='TB', fontname='Arial')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')

    # Input
    dot.node('iot_traffic', 'IoT Device\nNetwork Traffic',
             fillcolor='lightgreen', shape='cylinder')

    # Preprocessing
    with dot.subgraph(name='cluster_preprocess') as c:
        c.attr(label='Preprocessing Service', style='filled', fillcolor='lightgray')
        c.node('feature_extract', 'Feature Extraction\n(115 features)')
        c.node('normalize', 'Normalization')

    # Inference
    with dot.subgraph(name='cluster_inference') as c:
        c.attr(label='Inference Service', style='filled', fillcolor='lightgray')
        c.node('model_serve', 'TF Serving\n(Model Server)', fillcolor='yellow')
        c.node('anomaly_model', 'Anomaly Model', fillcolor='lightcoral')
        c.node('class_model', 'Classifier Model', fillcolor='lightcoral')

    # Post-processing
    with dot.subgraph(name='cluster_postprocess') as c:
        c.attr(label='Post-processing', style='filled', fillcolor='lightgray')
        c.node('threshold', 'Threshold Check')
        c.node('alert', 'Alert Generation')

    # Monitoring
    with dot.subgraph(name='cluster_monitoring') as c:
        c.attr(label='Monitoring', style='filled', fillcolor='lightgray')
        c.node('logs_mon', 'Logging\n(CloudWatch/ELK)')
        c.node('metrics_mon', 'Metrics\n(Prometheus)')
        c.node('dashboard', 'Dashboard\n(Grafana)')

    # Output
    dot.node('output_pred', 'Prediction\n(Benign/Malicious)',
             fillcolor='lightgreen', shape='note')
    dot.node('output_alert', 'Security Alert',
             fillcolor='red', shape='note')

    # Flow
    dot.edge('iot_traffic', 'feature_extract')
    dot.edge('feature_extract', 'normalize')
    dot.edge('normalize', 'model_serve')

    dot.edge('model_serve', 'anomaly_model', style='dashed')
    dot.edge('model_serve', 'class_model', style='dashed')

    dot.edge('anomaly_model', 'threshold')
    dot.edge('class_model', 'threshold')

    dot.edge('threshold', 'output_pred')
    dot.edge('threshold', 'alert', label='if malicious')
    dot.edge('alert', 'output_alert')

    # Monitoring connections
    dot.edge('model_serve', 'logs_mon', style='dotted')
    dot.edge('model_serve', 'metrics_mon', style='dotted')
    dot.edge('metrics_mon', 'dashboard', style='dashed')
    dot.edge('logs_mon', 'dashboard', style='dashed')

    return dot


def generate_file_structure_tree():
    """Generate file structure tree diagram."""
    dot = graphviz.Digraph(
        'file_structure',
        comment='Project File Structure'
    )
    dot.attr(rankdir='LR', fontname='Courier')
    dot.attr('node', shape='folder', style='filled', fillcolor='lightyellow')

    # Root
    dot.node('root', 'project/')

    # Main directories
    dot.node('src', 'src/')
    dot.node('tests', 'tests/')
    dot.node('docs', 'docs/')
    dot.node('data', 'data/')
    dot.node('configs', 'configs/')
    dot.node('scripts', 'scripts/')
    dot.node('outputs', 'outputs/')

    # src subdirectories
    dot.node('src_anomaly', 'anomaly_detection/')
    dot.node('src_class', 'classification/')
    dot.node('src_fl', 'federated_learning/')
    dot.node('src_data', 'data/')
    dot.node('src_common', 'common/')

    # Connections
    dot.edge('root', 'src')
    dot.edge('root', 'tests')
    dot.edge('root', 'docs')
    dot.edge('root', 'data')
    dot.edge('root', 'configs')
    dot.edge('root', 'scripts')
    dot.edge('root', 'outputs')

    dot.edge('src', 'src_anomaly')
    dot.edge('src', 'src_class')
    dot.edge('src', 'src_fl')
    dot.edge('src', 'src_data')
    dot.edge('src', 'src_common')

    return dot


def main():
    """Generate all target (2025) architecture diagrams."""
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(parents=True, exist_ok=True)

    diagrams = {
        'system_architecture': generate_system_architecture(),
        'data_pipeline': generate_data_pipeline(),
        'autoencoder_architecture': generate_autoencoder_architecture(),
        'classifier_architecture': generate_classifier_architecture(),
        'federated_architecture': generate_federated_architecture(),
        'deployment_architecture': generate_deployment_architecture(),
        'file_structure': generate_file_structure_tree(),
    }

    print("Generating target (2025) architecture diagrams...")
    for name, diagram in diagrams.items():
        # Save as PNG with target_ prefix
        png_path = output_dir / f'target_{name}'
        diagram.render(png_path, format='png', cleanup=True)
        print(f"  Generated: target_{name}.png")

        # Save Graphviz source (local only, not committed)
        gv_path = output_dir.parent / 'diagrams' / f'target_{name}.gv'
        diagram.save(gv_path)
        print(f"  Saved source: target_{name}.gv (local only)")

    print(f"\nAll diagrams generated in: {output_dir}")
    print("\nDiagrams created:")
    for name in diagrams.keys():
        print(f"  - {name}")


if __name__ == '__main__':
    main()
