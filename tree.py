import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("decision_tree")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'tree_built' not in st.session_state:
    st.session_state.tree_built = False
if 'build_logs' not in st.session_state:
    st.session_state.build_logs = []
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

@dataclass
class DecisionTreeConfig:
    """Configuration settings for the Decision Tree Builder."""
    min_samples_split: int = 2
    max_depth: Optional[int] = None
    min_information_gain: float = 0.0
    
    visualization: Dict[str, Any] = field(default_factory=lambda: {
        'output_format': 'png',
        'theme': 'modern',
        'dpi': 300
    })

    def validate(self) -> None:
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if not 0 <= self.min_information_gain <= 1:
            raise ValueError("min_information_gain must be between 0 and 1")

class TreeMetrics:
    def __init__(self):
        self.created_at = datetime.now()
        self.features = []
        self.target = ""
        self.node_count = 0
        self.max_depth = 0
        self.leaf_count = 0
        self.split_counts = {}
        self.feature_importance = {}
        self.build_time = 0

    def update_metrics(self, node_type: str, feature: str = None, depth: int = 0):
        if node_type == 'split':
            if feature:
                self.split_counts[feature] = self.split_counts.get(feature, 0) + 1
            self.node_count += 1
            self.max_depth = max(self.max_depth, depth)
        elif node_type == 'leaf':
            self.leaf_count += 1
            self.node_count += 1
            self.max_depth = max(self.max_depth, depth)

    def to_dict(self):
        return {
            'created_at': self.created_at.isoformat(),
            'features': self.features,
            'target': self.target,
            'node_count': self.node_count,
            'max_depth': self.max_depth,
            'leaf_count': self.leaf_count,
            'split_counts': self.split_counts,
            'feature_importance': self.feature_importance,
            'build_time': round(self.build_time, 3)
        }

class StreamlitTreeBuilder:
    def __init__(self):
        self.config = DecisionTreeConfig()
        self.metrics = TreeMetrics()
        self.build_logs = []
        self.dot = graphviz.Digraph()
        self.node_count = 0
        
    def log_message(self, message: str, level: str = 'info', indent: int = 0):
        formatted_message = f"{'  ' * indent}{message}"
        self.build_logs.append(formatted_message)

    def entropy(self, data: pd.DataFrame, target_column: str) -> tuple:
        values = data[target_column].value_counts(normalize=True)
        entropy_value = -np.sum(values * np.log2(values + np.finfo(float).eps))
        return entropy_value, dict(values)

    def information_gain(self, data: pd.DataFrame, split_column: str, target_column: str) -> float:
        total_entropy, _ = self.entropy(data, target_column)
        values = data[split_column].value_counts(normalize=True)
        counts = data[split_column].value_counts()
        weighted_entropy = 0
        
        self.log_message(f"Calculating Gain for feature '{split_column}':")
        
        for value in values.index:
            subset = data[data[split_column] == value]
            subset_entropy, _ = self.entropy(subset, target_column)
            weight = counts[value] / len(data)
            weighted_entropy += weight * subset_entropy
            self.log_message(f"- Value '{value}': entropy = {subset_entropy:.4f}", indent=1)

        gain = total_entropy - weighted_entropy
        self.log_message(f"Gain for feature '{split_column}': {gain:.4f}\n")
        return gain

    def best_split(self, data: pd.DataFrame, features: List[str], target_column: str) -> str:
        gains = {}
        for feature in features:
            gains[feature] = self.information_gain(data, feature, target_column)
        
        best_feature = max(gains, key=gains.get)
        self.log_message(f"⭐ Best feature: {best_feature} (Gain = {gains[best_feature]:.4f})\n")
        return best_feature

    def build_tree(self, data: pd.DataFrame, features: List[str], target_column: str) -> graphviz.Digraph:
        self.build_logs = []
        self.dot = graphviz.Digraph()
        self.dot.attr(rankdir='TB')
        self.node_count = 0
        
        self.metrics = TreeMetrics()
        self.metrics.features = features
        self.metrics.target = target_column
        
        start_time = time.time()
        self.log_message("Starting to build the decision tree...\n")
        
        class_dist = data[target_column].value_counts()
        total_samples = len(data)
        self.log_message("Initial dataset:")
        self.log_message(f"Total samples: {total_samples}", indent=1)
        for class_name, count in class_dist.items():
            percentage = (count / total_samples) * 100
            self.log_message(f"Class '{class_name}': {count} samples ({percentage:.1f}%)", indent=1)

        def build_tree_recursive(data: pd.DataFrame, features: List[str], parent_id: str = None, 
                               parent_value: str = None, depth: int = 0):
            node_id = f"node_{self.node_count}"
            self.node_count += 1
            
            if len(data[target_column].unique()) == 1:
                result = data[target_column].iloc[0]
                label = f"{target_column} = {result}"
                fillcolor = '#98FB98' if str(result).lower() == 'ya' else '#FFA07A'
                self.dot.node(node_id, label, shape='box', style='filled', fillcolor=fillcolor)
                if parent_id:
                    self.dot.edge(parent_id, node_id, label=str(parent_value))
                self.metrics.update_metrics('leaf', depth=depth)
                return
            
            if not features or (self.config.max_depth and depth >= self.config.max_depth):
                majority_class = data[target_column].mode()[0]
                class_counts = data[target_column].value_counts()
                label = f"{target_column} = {majority_class}\n"
                for cls, count in class_counts.items():
                    percentage = (count / len(data)) * 100
                    label += f"{cls}: {percentage:.1f}%\n"
                fillcolor = '#98FB98' if str(majority_class).lower() == 'ya' else '#FFA07A'
                self.dot.node(node_id, label, shape='box', style='filled', fillcolor=fillcolor)
                if parent_id:
                    self.dot.edge(parent_id, node_id, label=str(parent_value))
                self.metrics.update_metrics('leaf', depth=depth)
                return
            
            best_feature = self.best_split(data, features, target_column)
            self.dot.node(node_id, best_feature, shape='ellipse', style='filled', fillcolor='#C4A484')
            self.metrics.update_metrics('split', best_feature, depth)
            
            if parent_id:
                self.dot.edge(parent_id, node_id, label=str(parent_value))
            
            remaining_features = [f for f in features if f != best_feature]
            for value in sorted(data[best_feature].unique()):
                subset = data[data[best_feature] == value]
                if len(subset) > 0:
                    build_tree_recursive(subset, remaining_features, node_id, value, depth + 1)

        build_tree_recursive(data, features)
        
        self.metrics.build_time = time.time() - start_time
        self.log_message("\nDecision tree successfully built")
        
        return self.dot

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    
    return df

def show_dataset_info(df: pd.DataFrame, target_column: str, feature_columns: List[str]):
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### **Data Shape**")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")
    
    with col2:
        st.markdown("### **Columns**")
        st.write(f"Target: **{target_column}**")
        st.write(f"Features: **{', '.join(feature_columns)}**")

def show_parameter_values(df: pd.DataFrame):
    st.subheader("Parameter Values")
    st.write("### Unique Values for Each Column")
    for column in df.columns:
        unique_values = df[column].unique()
        st.write(f"**{column}**: {', '.join(map(str, unique_values))}")

def main():
    st.set_page_config(page_title="Decision Tree Builder", page_icon="🌳", layout="wide")
    st.title("🌳 **Decision Tree Builder**")
    
    uploaded_file = st.file_uploader("Upload Your Dataset (CSV format)", type="csv", label_visibility="collapsed")
    if uploaded_file:
        # Read and process data
        df = pd.read_csv(uploaded_file)
        df = process_data(df)
        st.session_state.data = df
        
        # Automatically set target as last column and features as all other columns
        target_column = df.columns[-1]
        feature_columns = df.columns[:-1].tolist()
        
        # Show dataset info
        show_dataset_info(df, target_column, feature_columns)
        
        # Show parameter values for each column
        show_parameter_values(df)
        
        st.sidebar.title("🎛 **Decision Tree Parameters**")
        
        max_depth = st.sidebar.number_input("Maximum Depth (0 for no limit)", min_value=0, value=3, step=1)
        min_samples_split = st.sidebar.number_input("Minimum Samples per Split", min_value=2, value=2, step=1)
        min_information_gain = st.sidebar.slider("Minimum Information Gain", 0.0, 1.0, 0.0, 0.01)
        
        # Add a nice separator in sidebar
        st.sidebar.markdown("---")
        
        # Build Tree Button
        if st.button("🔨 Build Decision Tree"):
            tree_builder = StreamlitTreeBuilder()
            tree_builder.config = DecisionTreeConfig(
                max_depth=max_depth if max_depth != 0 else None,
                min_samples_split=min_samples_split,
                min_information_gain=min_information_gain
            )
            
            st.session_state.tree_built = True
            dot = tree_builder.build_tree(df, feature_columns, target_column)
            st.session_state.metrics = tree_builder.metrics.to_dict()
            
            # Show Visualization
            st.subheader("Tree Visualization")
            st.graphviz_chart(dot)
            
            # Show Logs with scroll
            st.subheader("📝 Build Logs")
            log_container = st.container()
            with log_container:
                st.text_area("Logs", value="\n".join(tree_builder.build_logs), height=300, disabled=True)
            
            # Show Metrics
            st.subheader("📊 Tree Metrics")
            st.json(st.session_state.metrics)

if __name__ == "__main__":
    main()
