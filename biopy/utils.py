import pandas as pd
import umap
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from matplotlib.lines import Line2D
from sklearn.decomposition import PCA


def subset(df, metadata, condition):
    """Subset dataframe based on metadata condition."""
    metadata = metadata.loc[df.columns]
    assert metadata.index.equals(df.columns)
    idx = metadata.query(condition).index
    return df[idx]


def plot_umap_old(ax, x, metadata, colourbar=False, **kwargs):
    reducer = umap.UMAP()
    z = reducer.fit_transform(x.transpose())
    z = pd.DataFrame(
        z,
        index=x.columns,
        columns=['UMAP1', 'UMAP2']
    )
    z = z.join(metadata)
    ax = sns.scatterplot(
        data=z,
        x='UMAP1',
        y='UMAP2',
        edgecolor=None,
        ax=ax,
        **kwargs
    )

    if colourbar:
        import matplotlib.dates as mdates
        import matplotlib as mpl
        hue = kwargs.pop('hue', None)
        palette = kwargs.pop('palette', 'rocket')
        # Create normalization for colourbar
        norm = mpl.colors.Normalize(
            vmin=metadata[hue].min(),
            vmax=metadata[hue].max()
        )
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=palette)
        sm.set_array([])
        fig = ax.get_figure()
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(hue)
        cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    return ax


def plot_pca_old(ax, x, metadata, colourbar=False, **kwargs):
    '''PCA plot for visualisation of batch effects.'''
    pca = PCA(n_components=2)
    z = pca.fit_transform(x.transpose())
    var_ratio = pca.explained_variance_ratio_
    z = pd.DataFrame(
        z,
        index=x.columns,
        columns=['PC1', 'PC2']
    )
    z = z.join(metadata)
    ax = sns.scatterplot(
        data=z,
        x='PC1',
        y='PC2',
        edgecolor=None,
        ax=ax,
        **kwargs
    )
    ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)')

    if colourbar:
        hue = kwargs.pop('hue', None)
        palette = kwargs.pop('palette', 'rocket')
        # Create normalization for colourbar
        norm = mpl.colors.Normalize(
            vmin=metadata[hue].min(),
            vmax=metadata[hue].max()
        )
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=palette)
        sm.set_array([])
        fig = ax.get_figure()
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(hue)
        cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    return ax


def plot_pca(
    ax, x, metadata, colourbar=False, return_fig=True, **kwargs
):
    '''PCA plot for visualisation of batch effects.'''
    pca = PCA(n_components=2)
    z = pca.fit_transform(x.transpose())
    var_ratio = pca.explained_variance_ratio_
    z = pd.DataFrame(
        z,
        index=x.columns,
        columns=['PC1', 'PC2']
    )
    z = z.join(metadata)

    hue = kwargs.pop('hue', None)
    palette = kwargs.pop('palette', 'viridis')
    cmap_aliases = {'rocket': 'magma'}
    palette = cmap_aliases.get(palette, palette)
    kwargs.pop('legend', None)
    edgecolor = kwargs.pop('edgecolor', None)
    style = kwargs.pop('style', None)
    style_legend = kwargs.pop('style_legend', True)
    markers = ['o', 's', '^', 'D', 'X', 'v', '*', 'P', '<', '>', 'h', '8']
    scatter = None

    if style is not None and style in z.columns:
        style_values = z[style].fillna('Unknown').astype(str)
        unique_styles = pd.Index(style_values).unique()
        norm = None
        if hue is not None:
            norm = mpl.colors.Normalize(vmin=z[hue].min(), vmax=z[hue].max())
        handles = []
        for i, style_value in enumerate(unique_styles):
            mask = style_values == style_value
            marker = markers[i % len(markers)]
            if hue is not None:
                scatter = ax.scatter(
                    z.loc[mask, 'PC1'],
                    z.loc[mask, 'PC2'],
                    c=z.loc[mask, hue],
                    cmap=palette,
                    norm=norm,
                    marker=marker,
                    edgecolors=edgecolor,
                    **kwargs
                )
            else:
                scatter = ax.scatter(
                    z.loc[mask, 'PC1'],
                    z.loc[mask, 'PC2'],
                    marker=marker,
                    edgecolors=edgecolor,
                    **kwargs
                )
            handles.append(
                Line2D(
                    [0], [0],
                    marker=marker,
                    linestyle='',
                    color='black',
                    label=style_value
                )
            )
        if style_legend:
            ax.legend(handles=handles, title=style, loc='best')
    else:
        if hue is not None:
            scatter = ax.scatter(
                z['PC1'],
                z['PC2'],
                c=z[hue],
                cmap=palette,
                edgecolors=edgecolor,
                **kwargs
            )
        else:
            scatter = ax.scatter(
                z['PC1'],
                z['PC2'],
                edgecolors=edgecolor,
                **kwargs
            )

    ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)')

    if colourbar and hue is not None:
        fig = ax.get_figure()
        if style is not None and style in z.columns:
            norm = mpl.colors.Normalize(vmin=z[hue].min(), vmax=z[hue].max())
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=palette)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
        else:
            cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(hue)
        if 'date' in hue:
            cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    if return_fig:
        return ax
    else:
        print('Returning both axis and PCA coordinates for further use...')
        return (ax, z)


def plot_umap(
    ax, x, metadata, colourbar=False, return_fig=False, **kwargs
):
    reducer = umap.UMAP(n_components=2, random_state=42)
    z = reducer.fit_transform(x.transpose())
    z = pd.DataFrame(
        z,
        index=x.columns,
        columns=['UMAP1', 'UMAP2']
    )
    z = z.join(metadata)

    hue = kwargs.pop('hue', None)
    palette = kwargs.pop('palette', 'viridis')
    cmap_aliases = {'rocket': 'magma'}
    palette = cmap_aliases.get(palette, palette)
    kwargs.pop('legend', None)
    edgecolor = kwargs.pop('edgecolor', None)
    style = kwargs.pop('style', None)
    style_legend = kwargs.pop('style_legend', True)
    markers = ['o', 's', '^', 'D', 'X', 'v', '*', 'P', '<', '>', 'h', '8']
    scatter = None

    if style is not None and style in z.columns:
        style_values = z[style].fillna('Unknown').astype(str)
        unique_styles = pd.Index(style_values).unique()
        norm = None
        if hue is not None:
            norm = mpl.colors.Normalize(vmin=z[hue].min(), vmax=z[hue].max())
        handles = []
        for i, style_value in enumerate(unique_styles):
            mask = style_values == style_value
            marker = markers[i % len(markers)]
            if hue is not None:
                scatter = ax.scatter(
                    z.loc[mask, 'UMAP1'],
                    z.loc[mask, 'UMAP2'],
                    c=z.loc[mask, hue],
                    cmap=palette,
                    norm=norm,
                    marker=marker,
                    edgecolors=edgecolor,
                    **kwargs
                )
            else:
                scatter = ax.scatter(
                    z.loc[mask, 'UMAP1'],
                    z.loc[mask, 'UMAP2'],
                    marker=marker,
                    edgecolors=edgecolor,
                    **kwargs
                )
            handles.append(
                Line2D(
                    [0], [0],
                    marker=marker,
                    linestyle='',
                    color='black',
                    label=style_value
                )
            )
        if style_legend:
            ax.legend(handles=handles, title=style, loc='best')
    else:
        if hue is not None:
            scatter = ax.scatter(
                z['UMAP1'],
                z['UMAP2'],
                c=z[hue],
                cmap=palette,
                edgecolors=edgecolor,
                **kwargs
            )
        else:
            scatter = ax.scatter(
                z['UMAP1'],
                z['UMAP2'],
                edgecolors=edgecolor,
                **kwargs
            )

    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')

    if colourbar and hue is not None:
        fig = ax.get_figure()
        if style is not None and style in z.columns:
            norm = mpl.colors.Normalize(vmin=z[hue].min(), vmax=z[hue].max())
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=palette)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
        else:
            cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(hue)
        if 'date' in hue:
            cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    if return_fig:
        return ax
    else:
        print('Returning both axis and PCA coordinates for further use...')
        return (ax, z)
