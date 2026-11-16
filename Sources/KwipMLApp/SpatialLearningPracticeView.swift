import SwiftUI

/// Editable practice does not create an assessment receipt or supply imagined measurements.
enum SpatialLearningPracticeContent {
    enum Launcher: Hashable {
        case conv1d, conv2d, conv3d
        case transpose1d, transpose2d, transpose3d
        case depthwise, separable
        case bundled(String)
    }
    struct Entry: Identifiable {
        let id: String
        let title: String
        let launcher: Launcher
        let prediction: String
        let change: String
        let inspection: String
        var isAssessed: Bool { false }
    }
    static let contentVersion = 1
    static let authoredAt = "2026-10-08"
    static let disclosure = "Editable canvas practice, not scored assessment. These examples arrange synthetic features from tabular observations into lines, grids or volumes. They teach the operations; they are not recorded sound, photographs or scans, and their scores do not prove suitability for those tasks."
    static let terms = """
    **Window:** the neighboring positions examined together at one location. A convolution learns how much each position contributes; pooling uses a fixed summary such as the largest value or average.

    **Channel:** one separate stream of values at each position. For a real color image, red, green and blue can be channels. The example channels here are constructed features.

    **Stride:** how far the window moves between locations. A larger step often produces fewer positions in ordinary convolution or pooling.

    **Padding:** the boundary convention used outside the input. Convolution uses zeros here; max pooling ignores outside positions instead of treating them as zeros. Average pooling has an explicit choice about whether configured padding counts in the average.

    **Dilation:** the spacing between positions inside a window. Spacing of one examines neighboring positions; larger spacing reaches farther without adding more window values.

    **Batch:** several observations processed together. Window operations work on each observation separately; they do not mix people or examples together.

    **Shape:** the number of values along each direction. In these examples, read the directions as batch, channels, then length; or height and width; or depth, height and width. **Parameter:** a stored number the model can learn, such as a filter weight or bias. **Bias:** a learned offset added to an output.
    """
    static let runInstruction = "First run the unchanged example and inspect its actual output. Predict the effect of one change, then apply it and make another actual run. Keep the seed and other settings fixed, but do not assume trained values stay identical after an architecture change. If the graph reports an incompatible shape, read the shown dimensions and repair the downstream input width, or Undo the change. A refused or failed run has unavailable outputs; it is not an assessment failure."
    static let entries: [Entry] = [
        Entry(id: "conv1d", title: "Convolution along a line", launcher: .conv1d,
              prediction: "A learned window moves along each constructed feature line. Predict how a larger stride changes the number of output positions, separately from the number of output channels.",
              change: "Open the example and select Conv1d. Record window, stride and padding. Same spatial size requires stride 1: choose explicit padding first if you want to try a larger stride, then compare the displayed shape. Use the displayed output shape to keep the later flattened input compatible, or Undo if you want to preserve the original sample.",
              inspection: "Inspect the Conv1d input and output ports after each completed run. Compare length and channel count. Look at retained values only if capture is available; a few captured values do not describe every position."),
        Entry(id: "conv2d", title: "Convolution across a grid", launcher: .conv2d,
              prediction: "Height and width have separate window settings. Predict which output direction changes when you adjust only the width stride.",
              change: "Select the 2D Convolution block and read each pair as height, then width. Same spatial size requires stride [1,1]. To try a larger width stride, choose explicit padding first; then keep the height stride fixed. Check the displayed shape and downstream classifier input before running.",
              inspection: "Compare the captured grid shape before and after the change. Keep batch and channel directions separate from height and width. The constructed grid is not evidence that these tabular features are naturally neighbors."),
        Entry(id: "conv3d", title: "Convolution through a volume", launcher: .conv3d,
              prediction: "A volume has depth as well as height and width. Predict which dimensions change if only the depth stride increases.",
              change: "Select the 3D Convolution block and read its depth/height/width controls in that order. Same spatial size requires stride [1,1,1]; choose explicit padding before trying a larger depth stride. Keep height and width stride fixed, and check any downstream shape warning before the new run.",
              inspection: "Inspect the input and output shape, locating depth separately from channels. Compare the retained output coordinates if available. A small synthetic volume does not establish real scan or video performance."),
        transpose("conv_transpose1d", "Transposed convolution along a line", .transpose1d),
        transpose("conv_transpose2d", "Transposed convolution across a grid", .transpose2d),
        transpose("conv_transpose3d", "Transposed convolution through a volume", .transpose3d),
        Entry(id: "depthwise_conv2d", title: "Filter channels independently", launcher: .depthwise,
              prediction: "Depthwise convolution filters each input channel separately. Its multiplier is the number of different filters for each input channel; increasing it adds outputs without mixing input channels.",
              change: "Open the example, select the depthwise Convolution block and change the channel multiplier from two to one. Read the new output channel count and keep the downstream flattened input width compatible before running.",
              inspection: "Compare output channels and learned parameter count. Separate filters do not automatically combine information from different input channels; a later operation can do that."),
        Entry(id: "separable_conv2d", title: "Separate filtering from channel mixing", launcher: .separable,
              prediction: "This editable Block first filters each channel, then mixes channels using a one-position window. It is a two-part recipe with its own restrictions, rather than every possible ordinary convolution.",
              change: "Open the separable example and expand its container on the same canvas. Change its exposed multiplier from two to three. Follow the derived input width into the second convolution; preserve its output channel count.",
              inspection: "Compare both child shapes and the complete parameter count. The intermediate channel count can change while the final channel count stays fixed. If you release a derived connection, you become responsible for compatible child settings."),
        maximum("max_pool1d", "Largest value along a line"),
        maximum("max_pool2d", "Largest value across a grid"),
        maximum("max_pool3d", "Largest value through a volume"),
        Entry(id: "avg_pool1d", title: "Average neighboring values", launcher: .bundled("avg_pool1d"),
              prediction: "Average pooling summarizes a window without learning weights. Predict why including boundary padding in the count can change an edge average even when its real values are unchanged.",
              change: "Open the average-pooling example. Keep its window, stride and padding fixed, then toggle whether padding is included in the average. This change preserves the output shape, so compare it separately from changing the window size.",
              inspection: "Inspect edge positions and the unchanged output shape. The divisor is the number used to divide the sum: including configured padding can make it larger. Extra space used only to include a partial final window is not counted as configured padding. Inspect actual retained values rather than expecting identical trained weights.")
    ]
    private static func transpose(_ id: String, _ title: String, _ launcher: Launcher) -> Entry {
        Entry(id: id, title: title, launcher: launcher,
              prediction: "Transposed convolution spreads each input's learned contributions into output positions and adds overlaps. It is not an inverse that recovers information lost by an earlier convolution.",
              change: "Record the example's output shape, then reduce one output-padding setting where it is nonzero. Output padding selects the output extent; it does not append a row of zero values. Repair any downstream width mismatch before running, or Undo to return to the baseline.",
              inspection: "Compare output length, or the selected grid/volume direction, after the completed runs. Check that other directions remain as configured. Output size alone does not prove recovered information or improved prediction.")
    }
    private static func maximum(_ id: String, _ title: String) -> Entry {
        Entry(id: id, title: title, launcher: .bundled(id),
              prediction: "Max pooling keeps the largest value in each window, separately for each channel and observation. It adds no learned weights. Negative inputs still produce negative maxima if every real value in a window is negative.",
              change: "Open the example and record the window and stride. Change one stride direction, keeping the window fixed; an empty stride follows the window size. Check the new output shape and downstream input width before running.",
              inspection: "Compare the retained input/output shapes and available window values. If return indices is enabled, the indices identify winning positions; they are not another stream of learned scores. A boundary window does not compete against artificial zeros.")
    }
}

struct SpatialLearningPracticeView: View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store: WorkspaceStore
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            if !inLessonRail {
            LessonTextPanel(title: "Windows and channels · editable practice", text: SpatialLearningPracticeContent.disclosure)
            LessonTextPanel(title: "Words to know", text: SpatialLearningPracticeContent.terms)
            LessonTextPanel(title: "Predict → change → run → inspect", text: SpatialLearningPracticeContent.runInstruction)
            }
            ForEach(SpatialLearningPracticeContent.entries) { entry in
                VStack(alignment: .leading, spacing: 12) {
                    Text(entry.title).font(.system(size: 15, weight: .semibold))
                    if !inLessonRail {
                    LessonRichText(text: "**Predict**\n\(entry.prediction)\n\n**Change one setting**\n\(entry.change)\n\n**Inspect the result**\n\(entry.inspection)")
                    }
                    Button("Open editable example") { open(entry.launcher) }
                        .buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
                        .accessibilityLabel("Open " + entry.title + " example")
                }.padding(16).frame(maxWidth: .infinity, alignment: .leading)
                    .background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
                    .accessibilityIdentifier("learning.spatial-practice." + entry.id)
            }
            Text("Content v\(SpatialLearningPracticeContent.contentVersion) · Authored \(SpatialLearningPracticeContent.authoredAt) · Unscored practice")
                .font(.caption).foregroundStyle(MLTheme.muted)
        }
    }
    private func open(_ launcher: SpatialLearningPracticeContent.Launcher) {
        switch launcher {
        case .conv1d: _ = store.loadConv1dWorkflowExample()
        case .conv2d: _ = store.loadConv2dWorkflowExample()
        case .conv3d: _ = store.loadConv3dWorkflowExample()
        case .transpose1d: _ = store.loadConvTranspose1dWorkflowExample()
        case .transpose2d: _ = store.loadConvTranspose2dWorkflowExample()
        case .transpose3d: _ = store.loadConvTranspose3dWorkflowExample()
        case .depthwise: _ = store.loadDepthwiseConv2dWorkflowExample()
        case .separable: _ = store.loadSeparableConv2dWorkflowExample()
        case .bundled(let id): _ = store.loadComponentWorkflowExample(id)
        }
    }
}
