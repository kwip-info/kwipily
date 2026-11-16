import SwiftUI

struct SeparableConv2dLearningEntry: View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store: WorkspaceStore
    var body: some View {
        HStack(alignment: .top, spacing: 24) {
            VStack(alignment: .leading, spacing: 8) {
                if !inLessonRail {
                Text("Separable convolution inside an editable container").font(.title3.weight(.semibold))
                Text("Expand a container on the same canvas containing independent spatial filters followed by a 1×1 channel mixer. Its eleven exposed settings control both children; the intermediate channel count stays linked.")
                    .font(.system(size: 13)).foregroundStyle(MLTheme.muted)
                }
                DisclosureGroup("Understand factorization, channel mixing and independent weights") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("K[o,c,u,v] = Σₘ P[o,c·M+m,0,0] D[c·M+m,0,u,v]").font(.system(size: 12, design: .monospaced)).textSelection(.enabled)
                        Text("Depthwise kernels D filter each input channel independently. The pointwise matrix P mixes those responses at each spatial position. Both operations use cross-correlation; kernels are not reversed. There is no activation between these children, so their composition has the effective kernel K above.")
                        Text("b[o] = bP[o] + Σ꜀ Σₘ P[o,c·M+m,0,0] bD[c·M+m]").font(.system(size: 12, design: .monospaced)).textSelection(.enabled)
                        Text("Either bias can be disabled; its term becomes zero. Gradients reach both D and P through their product. For each input channel, the matrix of output kernels has rank at most M: this factorization restricts which dense kernels can be represented. It is not an arbitrary dense convolution with fewer stored numbers.")
                        Text("The example projects two measurements to [N,2,4,5]. With multiplier2 and kernel[2,3], depthwise produces [N,4,4,5] using 24 weights and four biases. Pointwise maps four channels to four using 16 weights and four biases. The Block has 48 parameters; the complete classifier has 330. Flatten still supplies 80 features to the classifier.")
                        Text("Select the container and change multiplier to3. The pointwise input updates to6; its output remains4, so the classifier still receives80 features. Total model parameters become352. Savings depend on these dimensions: increasing the multiplier can remove the advantage. Spatial settings belong to the depthwise stage; the pointwise kernel remains1×1 with stride1 and groups1.")
                        Text("Each Block instance owns independent weights. Exposed fields control their child targets; derived settings show their operands. Release a derived binding to keep its current default-context value and edit the target directly. Future parameter edits then require you to maintain compatible channel counts. Apply saves a private revision; existing sibling instances retain their pins.")
                        Text("Try adding a nonlinear activation between the children. The model can still train, but the single effective-kernel equation no longer describes it. ReLU in this bundled flow starts after the Block. These grids are constructed features from tabular observations, not source images; their spatial interpretation is a modeling choice.")
                    }.font(.caption).foregroundStyle(MLTheme.muted).padding(.top, 6)
                }
            }
            Spacer()
            Button("Open separable convolution example") { store.loadSeparableConv2dWorkflowExample() }.disabled(store.isBusy)
        }
    }
}
