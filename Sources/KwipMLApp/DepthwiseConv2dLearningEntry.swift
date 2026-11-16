import SwiftUI

/// Independent channel filtering explored through a prepared classifier.
struct DepthwiseConv2dLearningEntry: View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store: WorkspaceStore
    var body: some View {
        HStack(alignment: .top, spacing: 24) {
            VStack(alignment: .leading, spacing: 8) {
                if !inLessonRail {
                Text("Depthwise filter banks on a constructed grid").font(.title3.weight(.semibold))
                Text("Project two measurements into 40 features, reshape to [N,2,4,5], and learn two independent filters per input channel. ReLU and flatten expose 80 responses to the classifier.").font(.system(size: 13)).foregroundStyle(MLTheme.muted)
                }
                DisclosureGroup("Understand channel isolation and the multiplier") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Y[n,c·M+m,h,w] = b[c·M+m] + Σᵤ Σᵥ W[c·M+m,0,u,v] Xpad[n,c,h·sH+u·dH,w·sW+v·dW]").font(.system(size: 12, design: .monospaced)).textSelection(.enabled)
                        Text("M is the channel multiplier. Input channel c produces output channels c·M through c·M+M−1. Each has its own shared spatial kernel and optional bias; no sum crosses input channels. The operation is cross-correlation, so kernels are not reversed. Weight shape is [Cin·M,1,Kh,Kw].")
                        Text("Here Cin=2 and M=2 yield four outputs with W[4,1,2,3]: 24 weights and four optional biases. Projection, depthwise filtering and classifier total 310 parameters with bias. Multiplier1 means one filter per channel; increasing it adds different filters of the same channel, not channel mixing.")
                        Text("For each axis: Lout = floor((L + Pbefore + Pafter − d·(K−1) − 1)/s) + 1.").font(.system(size: 12, design: .monospaced)).textSelection(.enabled)
                        Text("Spatial pairs use [height,width]. Same padding with stride[1,1] preserves the 4×5 grid. The even height-2 kernel adds zero rows above and one below; width-3 adds one column on each side. Flatten converts [N,4,4,5] to [N,80]. Try multiplier3 for six output channels and update the classifier width to 120.")
                        Text("For a loss depending only on outputs c·M through c·M+M−1, gradients with respect to every other input channel and its depthwise kernels are zero. Changing one input channel cannot change another channel’s filter-bank outputs at this layer. Upstream projections and later layers can still couple the overall model.")
                        Text("To mix channels locally while retaining spatial positions, follow depthwise filtering with a Conv2d kernel[1,1], groups1 and matching input channels. That pointwise layer learns combinations across channels at each position. This example instead flattens and uses a Linear classifier, which mixes all spatial responses and channels.")
                        Text("The grid is constructed from ordinary measurements rather than source images. Real spatial data needs meaningful ordering, scaling and preparation. The authored depthwise component retains its own identity even though portable PyTorch code implements it with grouped Conv2d; changing component kind requires compatible checkpoints and refreshed caches.")
                    }.font(.caption).foregroundStyle(MLTheme.muted).padding(.top, 6)
                }
            }
            Spacer()
            Button("Open depthwise convolution example") { store.loadDepthwiseConv2dWorkflowExample() }.disabled(store.isBusy)
        }
    }
}
