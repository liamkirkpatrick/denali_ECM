#%% Import packages

# general
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import SpanSelector
import matplotlib.colors as mcolors

# my own modules
import sys
sys.path.append("../core_scripts/")
from ECMclass import ECM

#%% Load data
path_to_data = '../../data/processed_data_before-button-edits/'
path_to_new_data = '../data/processed_data/'
metadata_file = 'metadata.csv'
window = 10

print(path_to_data + metadata_file)

meta = pd.read_csv(path_to_data + metadata_file)

# this is where you select which core sections to proccess with the button editing tool. Currently set to run all dic1 sections
data = [
    ECM(row['core'], row['section'], row['face'], row['ACorDC'],path_to_data)
    for _, row in meta.iterrows()
    if row['core'] == 'dic1']

# import colormaps
cbar_2d = matplotlib.colormaps['Spectral']
cbar_line = matplotlib.colormaps['coolwarm']


# Print all of the commands for the user
print("################################################################")
print("################################################################")
print("/n")
print("\nWelcome to the ECM button editing tool!")
print("I apologize for the awkward interface, but I hit the point of diminishing returns on this one.")
print("You will be shown each run in sequence. If you want to reference parallel plots (i.e. want to see DC data when editing AC data), open the figures seperately.")
print("\nInstructions for button editing tool:")
print("\nCommands:")
print("  Select regions with mouse to toggle Button=1")
print("  'd' - delete last selection")
print("  'n' - next Y-dimension")
print("  'b' - back Y-dimension")
print("  'w' - write to file and move on")
print("  'r' - skip without saving and move on")
print("  'q' - quit program")
print("/n")
print("################################################################")
print("################################################################")


#%% Define processing function
def process_ecm(d):
    print(f"Plotting {d.core}, section {d.section}-{d.face}-{d.ACorDC}")


    # Extract raw vectors
    depth = d.depth
    y = d.y
    meas = d.meas
    if d.button_raw is not None:
        button_raw = d.button_raw
    else:
        button_raw = d.button
    
    d.smooth(window)
    depth_s = d.depth_s
    y_s = d.y_s
    meas_s = d.meas_s
    button_s = d.button_s

    # Initialize working Button column to zero
    button = np.zeros_like(d.button)


    # Build DataFrame
    df = pd.DataFrame({
        'True_depth(m)': depth,
        'Y_dimension(mm)': y,
        'meas': meas,
        'Button': button,
        'Button_raw': button_raw
    })

    df_s = pd.DataFrame({
        'True_depth(m)': depth_s,
        'Y_dimension(mm)': y_s,
        'meas': meas_s,
        'Button': button_s,
    })

    # Filename for saving
    fname = f"{d.core}-{d.section}-{d.face}-{d.ACorDC}.csv"

    # Unique Y-dimension values and colormap
    y_values = sorted(df['Y_dimension(mm)'].unique())
    cmap = mcolors.LinearSegmentedColormap.from_list('red_blue', ['red', 'blue'], N=256)
    norm = plt.Normalize(min(y_values), max(y_values))
    colors = {yv: cmap(norm(yv)) for yv in y_values}

    # Create figure with two subplots sharing the y-axis
    fig, (ax_left, ax_right) = plt.subplots(ncols=2, sharey=True, figsize=(10, 6))

    # left subplot
    ax_left.set_title("Top-down view")
    ax_left.set_ylabel('True_depth(m)')
    ax_left.set_xlabel('Distance Accross Core (mm)')
    #ax_left.invert_yaxis()

    # right subplot
    ax_right.invert_yaxis()
    ax_right.set_ylabel('True_depth(m)')
    ax_right.set_xlabel('meas')



    # ylimits
    for a in (ax_left, ax_right):
        a.set_ylim(df['True_depth(m)'].max()+0.1, df['True_depth(m)'].min()-0.1)

    # Plot each Y-dimension curve on the right subplot
    cnt = 0
    
    # replace flat plotting with capturing lines
    lines = {}
    for yv in y_values:
        subset = df_s[df_s['Y_dimension(mm)'] == yv]
        line, = ax_right.plot(subset['meas'], subset['True_depth(m)'], color=cbar_line(cnt/len(y_values)), label=str(yv))
        lines[yv] = line
        cnt += 1
    ax_right.legend(title='Y_dimension(mm)', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    # plot button points
    df_but = df_s.copy()
    mask = df_but['Button'] == 0
    df_but.loc[mask, 'meas'] = np.nan
    for yv in y_values:
        subset = df_but[df_but['Y_dimension(mm)'] == yv]
        if not subset['meas'].isna().all():  # Only plot if there are non-NaN values
            #print("    Plotting button points")
            line, ax_right.plot(subset['meas'], subset['True_depth(m)'], color='black', label='Button')

    # plot the left subplot
    if True:
        # Example vectors
        x_unique = np.unique(y_s)
        y_unique = np.unique(depth_s)
    
        # Create a meshgrid
        X, Y = np.meshgrid(x_unique, y_unique)
    
        # Map z to the grid and create a mask for button points
        Z = np.full(X.shape, np.nan)  # Initialize with NaNs
        button_mask = np.full(X.shape, False)  # Mask for button points
        for i in range(len(meas_s)):
            ix = np.where(x_unique == y_s[i])[0]
            iy = np.where(y_unique == depth_s[i])[0]
            Z[iy, ix] = meas_s[i]
            if button_s[i] == 1:
                button_mask[iy, ix] = True
    
        # Create a copy of the colormap
        cmap_with_black = cbar_2d.copy()
        
        # Plot using pcolormesh
        mesh = ax_left.pcolormesh(X, Y, Z, shading='auto', cmap=cbar_2d)
        
        # Overlay black points where button_s is 1
        ax_left.pcolormesh(X, Y, np.ma.masked_where(~button_mask, Z), 
                   shading='auto', cmap=plt.cm.colors.ListedColormap(['black']), 
                   alpha=0.5)


    # Data structures to track selections and patches
    selected_ranges = {yv: [] for yv in y_values}
    patches = {yv: [] for yv in y_values}
    current_idx = 0
    current_y = y_values[current_idx]
    fig.suptitle(f"{d.core}-{d.section}-{d.face}-{d.ACorDC} | Current Y: {current_y}", y=0.98)
    # Highlight the initial current Y line
    lines[current_y].set_linewidth(4)
    lines[current_y].set_zorder(2)

    # Callback for span selection
    def on_select(vmin, vmax):

        nonlocal current_y

        if vmax < vmin:
            vmin, vmax = vmax, vmin

        # Shade on both subplots
        pl = ax_left.axhspan(vmin, vmax, facecolor='gray', alpha=0.3)
        pr = ax_right.axhspan(vmin, vmax, facecolor='gray', alpha=0.3)
        patches[current_y].append((pl, pr))
        selected_ranges[current_y].append((vmin, vmax))

        # Update Button column for this Y-dimension
        mask = (
            (df['Y_dimension(mm)'] == current_y) &
            (df['True_depth(m)'] >= vmin) &
            (df['True_depth(m)'] <= vmax)
        )
        df.loc[mask, 'Button'] = 1
        fig.canvas.draw_idle()

    # Callback for key presses
    def on_key(event):
        nonlocal current_idx, current_y
        key = event.key
        
        if key == 'd':
            # Delete last selection for current Y
            if selected_ranges[current_y]:
                pl, pr = patches[current_y].pop()
                pl.remove(); pr.remove()
                selected_ranges[current_y].pop()
                # Recompute Button flags for remaining ranges
                df.loc[df['Y_dimension(mm)'] == current_y, 'Button'] = 0
                for a, b in selected_ranges[current_y]:
                    m2 = (
                        (df['Y_dimension(mm)'] == current_y) &
                        (df['True_depth(m)'] >= a) &
                        (df['True_depth(m)'] <= b)
                    )
                    df.loc[m2, 'Button'] = 1
                fig.canvas.draw_idle()

        elif key == 'n':
            # Advance to next Y-dimension
            for pl, pr in patches[current_y]:
                pl.remove(); pr.remove()
            patches[current_y].clear()
            if current_idx < len(y_values) - 1:
                current_idx += 1
                current_y = y_values[current_idx]
                # Reset all lines and highlight the new current Y line
                for yv, line in lines.items():
                    line.set_linewidth(0.5)
                    line.set_zorder(1)
                lines[current_y].set_linewidth(4)
                lines[current_y].set_zorder(2)
                fig.suptitle(f"{d.core}-{d.section}-{d.face}-{d.ACorDC} | Current Y: {current_y}", y=0.98)
                fig.canvas.draw_idle()
            else:
                print("    Last Y reached. Press 'w' to save or 'r' to skip.")

        elif key == 'b':
            # Advance to last Y-dimension
            for pl, pr in patches[current_y]:
                pl.remove(); pr.remove()
            patches[current_y].clear()
            if current_idx > 0:
                current_idx -= 1
                current_y = y_values[current_idx]
                # Reset all lines and highlight the new current Y line
                for yv, line in lines.items():
                    line.set_linewidth(0.5)
                    line.set_zorder(1)
                lines[current_y].set_linewidth(4)
                lines[current_y].set_zorder(2)
                fig.suptitle(f"{d.core}-{d.section}-{d.face}-{d.ACorDC} | Current Y: {current_y}", y=0.98)
                fig.canvas.draw_idle()
            else:
                print("    Can't go back from first Y.")

        elif key == 'w':
            # Save and move on
            df.to_csv(path_to_new_data + d.core + '/' + fname, index=False)
            print(f"    Saved {fname}")
            plt.close(fig)

        elif key == 'r':
            # Skip without saving
            print(f"    Skipped {fname}")
            plt.close(fig)
        elif key == 'q':
            # Quit the program
            ("    Quitting...")
            plt.close(fig)
            sys.exit()

    # Attach span selectors and keypress handler
    span_left = SpanSelector(ax_left, on_select, 'vertical',
                             useblit=True, props=dict(facecolor='gray', alpha=0.3),
                             interactive=False)
    span_right = SpanSelector(ax_right, on_select, 'vertical',
                              useblit=True, props=dict(facecolor='gray', alpha=0.3),
                              interactive=False)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

#%% Run on all ECM datasets
for d in data:
    process_ecm(d)
