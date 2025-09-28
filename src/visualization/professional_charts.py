"""
Professional Chart Generation System using Seaborn and Matplotlib
Creates publication-quality visualizations with advanced subplot layouts.
"""
import logging
import warnings
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import io
import base64

# Configure matplotlib and seaborn
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ProfessionalChartGenerator:
    """Generates publication-quality charts using seaborn and matplotlib."""

    def __init__(self):
        self.figure_size = (15, 10)
        self.dpi = 300
        self.font_family = 'DejaVu Sans'
        plt.rcParams['font.family'] = self.font_family
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.dpi'] = self.dpi

    def create_comprehensive_dashboard(self, data: pd.DataFrame,
                                     target_column: str = None) -> str:
        """
        Create a comprehensive analytical dashboard with multiple subplots.

        Args:
            data: Input dataframe for analysis
            target_column: Target variable for supervised learning analysis

        Returns:
            Base64 encoded image string
        """
        try:
            # Analyze data characteristics
            data_info = self._analyze_data_characteristics(data)

            # Create subplot layout based on data characteristics
            if data_info['is_classification'] and target_column:
                fig, axes = self._create_classification_dashboard(data, target_column)
            elif data_info['has_temporal_data']:
                fig, axes = self._create_temporal_dashboard(data)
            else:
                fig, axes = self._create_general_dashboard(data)

            # Apply professional styling
            self._apply_professional_styling(fig, axes)

            # Convert to base64
            return self._fig_to_base64(fig)

        except Exception as e:
            logger.error(f"Error creating comprehensive dashboard: {e}")
            return self._create_error_plot(str(e))

    def _analyze_data_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset characteristics to determine optimal visualization strategy."""
        info = {
            'numeric_columns': [],
            'categorical_columns': [],
            'temporal_columns': [],
            'is_classification': False,
            'has_temporal_data': False,
            'sample_size': len(data)
        }

        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                info['numeric_columns'].append(col)
            elif pd.api.types.is_datetime64_any_dtype(data[col]):
                info['temporal_columns'].append(col)
                info['has_temporal_data'] = True
            else:
                info['categorical_columns'].append(col)

        # Check if we have a classification target
        if info['categorical_columns']:
            # Assume first categorical column might be target for classification
            potential_target = info['categorical_columns'][0]
            unique_values = data[potential_target].nunique()
            if 2 <= unique_values <= 10:  # Reasonable range for classification
                info['is_classification'] = True

        return info

    def _create_classification_dashboard(self, data: pd.DataFrame,
                                       target_column: str) -> Tuple[plt.Figure, np.ndarray]:
        """Create dashboard optimized for classification problems."""

        # Determine layout based on number of features
        numeric_features = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col]) and col != target_column]

        if len(numeric_features) >= 4:
            fig, axes = plt.subplots(3, 2, figsize=self.figure_size)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
            axes = axes.flatten()

        # 1. Target distribution
        self._plot_target_distribution(data, target_column, axes[0])

        # 2. Feature correlation heatmap
        if len(numeric_features) > 1:
            correlation_matrix = data[numeric_features].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',
                       center=0, ax=axes[1], fmt='.2f')
            axes[1].set_title('Feature Correlation Matrix')
        else:
            axes[1].text(0.5, 0.5, 'Insufficient numeric features\nfor correlation analysis',
                        ha='center', va='center', transform=axes[1].transAxes)

        # 3. Feature distributions by target
        if len(numeric_features) > 0:
            feature = numeric_features[0]
            self._plot_feature_by_target(data, feature, target_column, axes[2])

        # 4. Box plots for numeric features
        if len(numeric_features) > 0:
            self._plot_box_plots(data, numeric_features[:3], axes[3] if len(axes) > 3 else None)

        # Hide unused subplots
        for i in range(len(numeric_features) + 2, len(axes)):
            if i < len(axes):
                axes[i].set_visible(False)

        plt.tight_layout()
        return fig, axes

    def _create_temporal_dashboard(self, data: pd.DataFrame) -> Tuple[plt.Figure, np.ndarray]:
        """Create dashboard optimized for temporal/time-series data."""

        fig, axes = plt.subplots(3, 2, figsize=self.figure_size)
        axes = axes.flatten()

        temporal_cols = [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]
        numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]

        if temporal_cols and numeric_cols:
            # Time series plot
            time_col = temporal_cols[0]
            value_col = numeric_cols[0]

            sns.lineplot(data=data, x=time_col, y=value_col, ax=axes[0])
            axes[0].set_title('Time Series Trend')
            axes[0].tick_params(axis='x', rotation=45)

            # Seasonal decomposition (if enough data)
            if len(data) > 50:
                self._plot_seasonal_decomposition(data, time_col, value_col, axes[1])

            # Temporal heatmap (if categorical data exists)
            categorical_cols = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])
                              and not pd.api.types.is_datetime64_any_dtype(data[col])]

            if categorical_cols:
                self._plot_temporal_heatmap(data, time_col, categorical_cols[0], axes[2])

        # Additional plots
        if len(numeric_cols) > 1:
            self._plot_scatter_matrix(data, numeric_cols[:4], axes[4:])
        else:
            axes[4].text(0.5, 0.5, 'Limited temporal analysis\navailable with single metric',
                        ha='center', va='center', transform=axes[4].transAxes)

        plt.tight_layout()
        return fig, axes

    def _create_general_dashboard(self, data: pd.DataFrame) -> Tuple[plt.Figure, np.ndarray]:
        """Create general-purpose analytical dashboard."""

        fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
        axes = axes.flatten()

        numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]

        # Distribution plots
        if numeric_cols:
            for i, col in enumerate(numeric_cols[:4]):
                if i < len(axes):
                    sns.histplot(data[col], kde=True, ax=axes[i])
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].axvline(data[col].mean(), color='red', linestyle='--', label='Mean')
                    axes[i].axvline(data[col].median(), color='green', linestyle='--', label='Median')
                    axes[i].legend()

        plt.tight_layout()
        return fig, axes

    def _plot_target_distribution(self, data: pd.DataFrame, target_column: str, ax: plt.Axes):
        """Plot target variable distribution for classification."""
        value_counts = data[target_column].value_counts()

        # Create horizontal bar plot for better readability
        bars = ax.barh(range(len(value_counts)), value_counts.values)

        # Color bars based on values
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.Set3(i / len(value_counts)))

        ax.set_yticks(range(len(value_counts)))
        ax.set_yticklabels(value_counts.index)
        ax.set_xlabel('Count')
        ax.set_title(f'Distribution of {target_column}')
        ax.grid(axis='x', alpha=0.3)

        # Add percentage labels
        for i, v in enumerate(value_counts.values):
            ax.text(v + 0.5, i, f'{v/len(data)*100".1f"}%', va='center')

    def _plot_feature_by_target(self, data: pd.DataFrame, feature: str,
                               target: str, ax: plt.Axes):
        """Plot feature distribution by target class."""
        try:
            sns.boxplot(data=data, x=target, y=feature, ax=ax)
            ax.set_title(f'{feature} by {target}')
            ax.tick_params(axis='x', rotation=45)

            # Add statistical annotations
            from scipy.stats import f_oneway
            classes = data[target].unique()
            if len(classes) == 2:
                # T-test for binary classification
                group1 = data[data[target] == classes[0]][feature]
                group2 = data[data[target] == classes[1]][feature]

                if len(group1) > 1 and len(group2) > 1:
                    from scipy.stats import ttest_ind
                    t_stat, p_value = ttest_ind(group1, group2)

                    ax.text(0.02, 0.98, f'p-value: {p_value".3f"}',
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except Exception as e:
            logger.warning(f"Could not create feature-by-target plot: {e}")
            ax.text(0.5, 0.5, 'Feature analysis\nnot available',
                   ha='center', va='center', transform=ax.transAxes)

    def _plot_box_plots(self, data: pd.DataFrame, features: List[str], ax: plt.Axes):
        """Create comparative box plots for multiple features."""
        try:
            plot_data = data[features].melt()
            sns.boxplot(data=plot_data, x='variable', y='value', ax=ax)
            ax.set_title('Feature Distribution Comparison')
            ax.tick_params(axis='x', rotation=45)
        except Exception as e:
            logger.warning(f"Could not create box plots: {e}")
            ax.text(0.5, 0.5, 'Box plot comparison\nnot available',
                   ha='center', va='center', transform=ax.transAxes)

    def _plot_seasonal_decomposition(self, data: pd.DataFrame, time_col: str,
                                   value_col: str, ax: plt.Axes):
        """Plot seasonal decomposition of time series."""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            # Ensure data is sorted by time
            ts_data = data.sort_values(time_col).set_index(time_col)[value_col]

            if len(ts_data) > 50:  # Need sufficient data
                decomposition = seasonal_decompose(ts_data, model='additive', period=min(24, len(ts_data)//3))

                # Plot decomposition
                ax.plot(decomposition.trend, label='Trend', linewidth=2)
                ax.plot(decomposition.seasonal, label='Seasonal', alpha=0.7)
                ax.plot(decomposition.residual, label='Residual', alpha=0.5)
                ax.set_title('Seasonal Decomposition')
                ax.legend()
                ax.tick_params(axis='x', rotation=45)
        except Exception as e:
            logger.warning(f"Could not create seasonal decomposition: {e}")
            ax.text(0.5, 0.5, 'Seasonal analysis\nnot available',
                   ha='center', va='center', transform=ax.transAxes)

    def _plot_temporal_heatmap(self, data: pd.DataFrame, time_col: str,
                              category_col: str, ax: plt.Axes):
        """Create a heatmap showing temporal patterns by category."""
        try:
            # Group by time and category
            heatmap_data = data.groupby([pd.Grouper(key=time_col, freq='D'), category_col]).size().unstack()

            if not heatmap_data.empty:
                sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax)
                ax.set_title(f'Temporal Pattern by {category_col}')
                ax.tick_params(axis='x', rotation=45)
        except Exception as e:
            logger.warning(f"Could not create temporal heatmap: {e}")
            ax.text(0.5, 0.5, 'Temporal heatmap\nnot available',
                   ha='center', va='center', transform=ax.transAxes)

    def _plot_scatter_matrix(self, data: pd.DataFrame, features: List[str], axes: List[plt.Axes]):
        """Create scatter plot matrix for feature relationships."""
        try:
            n_features = min(len(features), len(axes))

            for i in range(n_features):
                for j in range(n_features):
                    if i != j and i * n_features + j < len(axes):
                        ax = axes[i * n_features + j]

                        if i < len(features) and j < len(features):
                            sns.scatterplot(data=data, x=features[j], y=features[i], ax=ax, alpha=0.6)
                            ax.set_xlabel(features[j])
                            ax.set_ylabel(features[i])

                            # Add correlation coefficient
                            if len(data) > 10:
                                corr = data[features[i]].corr(data[features[j]])
                                ax.text(0.05, 0.95, f'r={corr".2f"}',
                                       transform=ax.transAxes, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except Exception as e:
            logger.warning(f"Could not create scatter matrix: {e}")

    def _apply_professional_styling(self, fig: plt.Figure, axes: np.ndarray):
        """Apply professional styling to the entire figure."""

        # Set overall theme
        sns.set_style("whitegrid", {
            'grid.linestyle': '--',
            'grid.color': '#E5E5E5',
            'axes.facecolor': '#FAFAFA'
        })

        # Configure figure background
        fig.patch.set_facecolor('#FAFAFA')

        # Style each subplot
        for ax in axes.flat:
            if ax.get_visible():
                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # Light grid
                ax.grid(True, alpha=0.3, linestyle='--')

                # Professional tick styling
                ax.tick_params(axis='both', which='major', labelsize=9, direction='out')

        # Add timestamp
        fig.text(0.99, 0.01, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                ha='right', va='bottom', alpha=0.7, fontsize=8)

        plt.tight_layout()

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string."""
        try:
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=self.dpi, bbox_inches='tight',
                       facecolor=fig.get_facecolor(), edgecolor='none')
            img_buffer.seek(0)

            img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            plt.close(fig)  # Close to free memory

            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Error converting figure to base64: {e}")
            plt.close(fig)
            return ""

    def _create_error_plot(self, error_message: str) -> str:
        """Create an error plot when visualization fails."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.text(0.5, 0.5, f'Visualization Error:\n{error_message}',
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='#FFE6E6', alpha=0.8),
               fontsize=12, color='#D8000C')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        return self._fig_to_base64(fig)

    def create_statistical_summary_plot(self, data: pd.DataFrame) -> str:
        """Create a comprehensive statistical summary visualization."""

        try:
            numeric_data = data.select_dtypes(include=[np.number])

            if numeric_data.empty:
                return self._create_error_plot("No numeric data available for statistical summary")

            # Create subplot layout
            n_features = len(numeric_data.columns)
            cols = min(3, n_features)
            rows = (n_features + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
            if rows == 1:
                axes = axes.reshape(1, -1) if cols > 1 else [[axes]]
            axes = axes.flatten()

            # Create plots for each numeric feature
            for i, column in enumerate(numeric_data.columns):
                if i < len(axes):
                    ax = axes[i]

                    # Distribution plot with statistical annotations
                    sns.histplot(numeric_data[column], kde=True, ax=ax, alpha=0.7)

                    # Add statistical annotations
                    mean_val = numeric_data[column].mean()
                    median_val = numeric_data[column].median()
                    std_val = numeric_data[column].std()

                    stats_text = f'μ={mean_val:.2f}\nσ={std_val:.2f}\nmedian={median_val:.2f}'
                    ax.text(0.7, 0.8, stats_text, transform=ax.transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                           verticalalignment='top', fontsize=9)

                    ax.set_title(f'{column} Distribution')
                    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label='Mean')
                    ax.axvline(median_val, color='green', linestyle='--', alpha=0.7, label='Median')
                    ax.legend()

            # Hide unused subplots
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)

            self._apply_professional_styling(fig, axes)
            return self._fig_to_base64(fig)

        except Exception as e:
            logger.error(f"Error creating statistical summary: {e}")
            return self._create_error_plot(str(e))

    def create_correlation_heatmap(self, data: pd.DataFrame,
                                 method: str = 'pearson') -> str:
        """Create an advanced correlation heatmap with significance testing."""

        try:
            numeric_data = data.select_dtypes(include=[np.number])

            if numeric_data.empty or len(numeric_data.columns) < 2:
                return self._create_error_plot("Insufficient numeric data for correlation analysis")

            # Calculate correlation matrix
            corr_matrix = numeric_data.corr(method=method)

            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))

            # Create custom colormap
            cmap = sns.diverging_palette(220, 20, as_cmap=True)

            # Create heatmap
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap=cmap,
                       center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                       ax=ax, fmt='.2f', annot_kws={"size": 8})

            ax.set_title(f'Correlation Matrix ({method.title()})', fontsize=14, pad=20)

            # Add significance indicators (simplified)
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:  # Strong correlation
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                                 edgecolor='red', lw=2, alpha=0.7))

            plt.tight_layout()
            return self._fig_to_base64(fig)

        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            return self._create_error_plot(str(e))

    def create_entrepreneurship_dashboard(self, students_data: pd.DataFrame,
                                        alumni_data: pd.DataFrame) -> str:
        """Create specialized dashboard for entrepreneurship survey analysis."""

        try:
            fig = plt.figure(figsize=(20, 12))

            # Create grid layout
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            # 1. Top-left: Student demographics
            ax1 = fig.add_subplot(gs[0, 0])
            if 'Gender' in students_data.columns:
                gender_counts = students_data['Gender'].value_counts()
                colors = sns.color_palette('Set2')
                wedges, texts, autotexts = ax1.pie(gender_counts.values,
                                                  labels=gender_counts.index,
                                                  autopct='%1.1f%%', colors=colors)
                ax1.set_title('Student Gender Distribution')

            # 2. Top-middle: Academic levels
            ax2 = fig.add_subplot(gs[0, 1])
            if 'Academic Level' in students_data.columns:
                level_counts = students_data['Academic Level'].value_counts()
                sns.barplot(x=level_counts.values, y=level_counts.index, ax=ax2, palette='viridis')
                ax2.set_title('Academic Level Distribution')
                ax2.set_xlabel('Count')

            # 3. Top-right: Employment status
            ax3 = fig.add_subplot(gs[0, 2])
            if 'Employment Status' in students_data.columns:
                emp_counts = students_data['Employment Status'].value_counts()
                sns.barplot(x=emp_counts.index, y=emp_counts.values, ax=ax3, palette='Set3')
                ax3.set_title('Employment Status')
                ax3.tick_params(axis='x', rotation=45)

            # 4. Bottom-left: Response patterns (sample questions)
            ax4 = fig.add_subplot(gs[1, :2])
            sample_questions = [col for col in students_data.columns
                              if col.startswith(('Entrepreneurial', 'Exposure', 'Taught', 'Learned', 'Networking'))][:6]

            if sample_questions:
                # Convert categorical responses to numeric for visualization
                response_mapping = {'SD': 1, 'D': 2, 'A': 3, 'SA': 4}
                plot_data = students_data[sample_questions].replace(response_mapping)

                sns.boxplot(data=plot_data, ax=ax4)
                ax4.set_title('Response Patterns Across Key Questions')
                ax4.set_ylabel('Agreement Level (1=SD, 4=SA)')
                ax4.tick_params(axis='x', rotation=45)

            # 5. Bottom-right: Alumni/Staff insights
            ax5 = fig.add_subplot(gs[2, :])
            if 'Status' in alumni_data.columns and 'Employment/Business Sector' in alumni_data.columns:
                sector_by_status = pd.crosstab(alumni_data['Status'], alumni_data['Employment/Business Sector'])
                sns.heatmap(sector_by_status, annot=True, fmt='d', cmap='YlGnBu', ax=ax5)
                ax5.set_title('Employment Sector by Stakeholder Status')

            # Add overall title
            fig.suptitle('University of Abuja Entrepreneurship Education Analysis Dashboard',
                        fontsize=16, y=0.98)

            plt.tight_layout()
            return self._fig_to_base64(fig)

        except Exception as e:
            logger.error(f"Error creating entrepreneurship dashboard: {e}")
            return self._create_error_plot(str(e))

# Global chart generator instance
chart_generator = ProfessionalChartGenerator()
