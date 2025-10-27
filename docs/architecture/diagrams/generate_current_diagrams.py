"""
Current System Architecture Diagrams (As-Is - 2020 Version)

This script generates diagrams showing the CURRENT state of the system
before modernization. This documents the 2020-era architecture.
"""

import graphviz
from pathlib import Path


def generate_current_system_architecture():
    """Generate current (2020) system architecture diagram."""
    dot = graphviz.Digraph(
        'current_system_architecture',
        comment='Current IoT Botnet Detection System (2020)'
    )
    dot.attr(rankdir='TB', splines='ortho', fontname='Arial')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')

    # Data Layer
    with dot.subgraph(name='cluster_data') as c:
        c.attr(label='Data Layer (Current)', style='filled', fillcolor='lightgray')
        c.node('dataset', 'N-BaIoT Dataset\n(UCI)')
        c.node('download', 'scripts/download_data.py')
        c.node('storage', 'Data Storage\n(CSV Files)\n❗ Mixed locations')

    # Preprocessing (current - has issues)
    with dot.subgraph(name='cluster_preprocessing') as c:
        c.attr(label='Preprocessing (Current)', style='filled', fillcolor='lightgray')
        c.node('loader', 'Data Loader\n❗ Hard-coded paths')
        c.node('fisher', 'Fisher Score\n(data/fisher/)')
        c.node('normalization', 'StandardScaler\n❌ DATA LEAKAGE\nfits on train+val', fillcolor='#ffcccc')

    # Models (current)
    with dot.subgraph(name='cluster_models') as c:
        c.attr(label='Model Layer (Current)', style='filled', fillcolor='lightgray')
        c.node('anomaly', 'Anomaly Detection\nanomal-detection/\n❗ Hyphenated name', fillcolor='lightgreen')
        c.node('classifier', 'Classification\nclassification/', fillcolor='lightgreen')
        c.node('federated', 'Federated Learning\nTFF (broken)\n❌ Never used botnet data', fillcolor='#ffcccc')

    # Evaluation
    with dot.subgraph(name='cluster_evaluation') as c:
        c.attr(label='Evaluation (Current)', style='filled', fillcolor='lightgray')
        c.node('metrics', 'Metrics\n❗ Print statements')
        c.node('lime', 'LIME\n✅ Working')

    # Output (current - disorganized)
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Output (Current)', style='filled', fillcolor='lightgray')
        c.node('models_out', 'Models\n❗ Mixed locations\n.h5 format')
        c.node('logs', 'Logs\n❗ In module dirs')

    # Connections
    dot.edge('dataset', 'download')
    dot.edge('download', 'storage')
    dot.edge('storage', 'loader')
    dot.edge('loader', 'fisher')
    dot.edge('fisher', 'normalization')

    dot.edge('normalization', 'anomaly')
    dot.edge('normalization', 'classifier')
    dot.edge('normalization', 'federated')

    dot.edge('anomaly', 'metrics')
    dot.edge('classifier', 'metrics')
    dot.edge('classifier', 'lime')

    dot.edge('anomaly', 'models_out')
    dot.edge('classifier', 'models_out')
    dot.edge('anomaly', 'logs')
    dot.edge('classifier', 'logs')

    # Add legend
    with dot.subgraph(name='cluster_legend') as c:
        c.attr(label='Legend', style='filled', fillcolor='white')
        c.node('legend_ok', '✅ Working', fillcolor='lightgreen', shape='note')
        c.node('legend_warn', '❗ Issues', fillcolor='lightyellow', shape='note')
        c.node('legend_error', '❌ Critical Bug', fillcolor='#ffcccc', shape='note')

    return dot


def generate_current_file_structure():
    """Generate current file structure diagram."""
    dot = graphviz.Digraph(
        'current_file_structure',
        comment='Current Project Structure (2020)'
    )
    dot.attr(rankdir='LR', fontname='Courier', fontsize='10')
    dot.attr('node', shape='folder', style='filled', fillcolor='lightyellow')

    # Root
    dot.node('root', 'project/')

    # Main directories (current)
    dot.node('anomaly', 'anomaly-detection/\n❗ Hyphenated', fillcolor='#ffeecc')
    dot.node('class', 'classification/')
    dot.node('jupyter', 'jupyter/')
    dot.node('scripts', 'scripts/')
    dot.node('config', 'config/')
    dot.node('data', 'data/')
    dot.node('docs', 'docs/')
    dot.node('analysis', 'analysis/')

    # Files in anomaly-detection (mixed concerns)
    dot.node('anomaly_train', 'train_autoencoder.py', shape='note', fillcolor='lightblue')
    dot.node('anomaly_test', 'test_autoencoder.py', shape='note', fillcolor='lightblue')
    dot.node('anomaly_models', 'models/\n❗ In module dir', shape='folder', fillcolor='#ffeecc')
    dot.node('anomaly_logs', 'logs/\n❗ In module dir', shape='folder', fillcolor='#ffeecc')
    dot.node('anomaly_lime', 'lime/\n❗ In module dir', shape='folder', fillcolor='#ffeecc')

    # Connections
    dot.edge('root', 'anomaly')
    dot.edge('root', 'class')
    dot.edge('root', 'jupyter')
    dot.edge('root', 'scripts')
    dot.edge('root', 'config')
    dot.edge('root', 'data')
    dot.edge('root', 'docs')
    dot.edge('root', 'analysis')

    dot.edge('anomaly', 'anomaly_train')
    dot.edge('anomaly', 'anomaly_test')
    dot.edge('anomaly', 'anomaly_models')
    dot.edge('anomaly', 'anomaly_logs')
    dot.edge('anomaly', 'anomaly_lime')

    # Add note
    dot.node('note', 'Issues:\n- Hyphenated names\n- Mixed concerns\n- No tests/\n- No src/ layout',
             shape='note', fillcolor='#ffcccc')

    return dot


def generate_current_data_pipeline():
    """Generate current data pipeline with data leakage highlighted."""
    dot = graphviz.Digraph(
        'current_data_pipeline',
        comment='Current Data Pipeline (with data leakage)'
    )
    dot.attr(rankdir='LR', fontname='Arial')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')

    # Pipeline stages
    dot.node('raw', 'Raw CSV Files', fillcolor='lightyellow')
    dot.node('load', 'Load CSV\npandas.read_csv()')
    dot.node('combine', 'Combine Devices\n❗ Using .append()', fillcolor='#ffeecc')
    dot.node('label', 'Add Labels')
    dot.node('split', 'Train/Val/Test Split\n(70/15/15)')
    dot.node('scale', '❌ DATA LEAKAGE\nscaler.fit(train.append(val))', fillcolor='#ffcccc')
    dot.node('transform', 'Transform All Sets')
    dot.node('ready', 'Training Ready', fillcolor='lightgreen')

    # Flow
    dot.edge('raw', 'load')
    dot.edge('load', 'combine')
    dot.edge('combine', 'label')
    dot.edge('label', 'split')
    dot.edge('split', 'scale', label='train + val ❌')
    dot.edge('scale', 'transform')
    dot.edge('transform', 'ready')

    # Correct way (for comparison)
    dot.node('correct', 'CORRECT:\nscaler.fit(train only)', fillcolor='lightgreen', shape='note')
    dot.edge('correct', 'scale', style='dashed', label='should be')

    return dot


def generate_current_dependencies():
    """Generate current dependency diagram."""
    dot = graphviz.Digraph(
        'current_dependencies',
        comment='Current Dependencies (2020-era)'
    )
    dot.attr(rankdir='TB', fontname='Arial')
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue')

    # Core dependencies
    dot.node('python', 'Python 3.8', fillcolor='lightyellow')

    # TensorFlow stack
    dot.node('tf', 'TensorFlow 2.10', fillcolor='#ffeecc')
    dot.node('keras_tf', 'tensorflow.keras', fillcolor='lightgreen')
    dot.node('keras_standalone', 'keras (standalone)\n❗ Mixed imports', fillcolor='#ffcccc')
    dot.node('tff', 'TensorFlow Federated 0.40\n❌ Broken dependencies', fillcolor='#ffcccc')

    # Data stack
    dot.node('pandas', 'Pandas 1.3.5\n❗ Uses .append()', fillcolor='#ffeecc')
    dot.node('numpy', 'NumPy 1.21', fillcolor='lightgreen')
    dot.node('sklearn', 'scikit-learn 1.0', fillcolor='lightgreen')

    # Others
    dot.node('lime_lib', 'LIME 0.2.0', fillcolor='lightgreen')

    # Dependencies
    dot.edge('python', 'tf')
    dot.edge('tf', 'keras_tf')
    dot.edge('tf', 'keras_standalone', style='dashed', label='mixed')
    dot.edge('tf', 'tff')
    dot.edge('python', 'pandas')
    dot.edge('python', 'numpy')
    dot.edge('python', 'sklearn')
    dot.edge('python', 'lime_lib')

    # Issues
    with dot.subgraph(name='cluster_issues') as c:
        c.attr(label='Known Issues', style='filled', fillcolor='white')
        c.node('issue1', 'Deprecated pandas.append()', shape='note', fillcolor='#ffcccc')
        c.node('issue2', 'Mixed keras imports', shape='note', fillcolor='#ffcccc')
        c.node('issue3', 'TFF impossible to install', shape='note', fillcolor='#ffcccc')

    return dot


def main():
    """Generate all current system diagrams."""
    output_dir = Path(__file__).parent.parent / 'images' / 'current'
    output_dir.mkdir(parents=True, exist_ok=True)

    diagrams = {
        'system_architecture': generate_current_system_architecture(),
        'file_structure': generate_current_file_structure(),
        'data_pipeline': generate_current_data_pipeline(),
        'dependencies': generate_current_dependencies(),
    }

    print("Generating CURRENT system architecture diagrams...")
    print(f"Output directory: {output_dir}")

    for name, diagram in diagrams.items():
        # Save as PNG with current_ prefix
        png_path = output_dir / f'current_{name}'
        diagram.render(png_path, format='png', cleanup=True)
        print(f"  [OK] Generated: current_{name}.png")

        # Save Graphviz source (local only, not committed)
        gv_path = output_dir.parent.parent / 'diagrams' / f'current_{name}.gv'
        diagram.save(gv_path)
        print(f"  [OK] Saved source: current_{name}.gv (local only)")

    print(f"\n[SUCCESS] All CURRENT system diagrams generated in: {output_dir}")
    print("\nDiagrams created:")
    for name in diagrams.keys():
        print(f"  - {name}")

    print("\nThese diagrams show the 2020-era system BEFORE modernization.")
    print("They highlight existing issues like data leakage, deprecated code, etc.")


if __name__ == '__main__':
    main()
