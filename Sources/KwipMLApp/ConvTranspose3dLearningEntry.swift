import SwiftUI

/// Volume scatter geometry backed by a runnable prepared classifier.
struct ConvTranspose3dLearningEntry: View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store: WorkspaceStore
    var body: some View {
        HStack(alignment: .top, spacing: 24) {
            VStack(alignment: .leading, spacing: 8) {
                if !inLessonRail {
                Text("Transposed convolution on a constructed volume").font(.title3.weight(.semibold))
                Text("Project two measurements into 48 features, reshape to [N,2,2,3,4], and expand with grouped volume kernels to [N,4,3,5,7]. ReLU and flatten expose 420 responses to the classifier.").font(.system(size: 13)).foregroundStyle(MLTheme.muted)
                }
                DisclosureGroup("Understand volume scatter, overlap and output geometry") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Y[n,g·(Cout/G)+o,z·sD−pD+u·dD,y·sH−pH+v·dH,x·sW−pW+w·dW] += X[n,i,z,y,x] W[i,o,u,v,w]").font(.system(size: 12, design: .monospaced)).textSelection(.enabled)
                        Text("Start at zero, scatter each weighted input into valid volume coordinates, sum overlaps, then add one bias per output channel. Input channel i belongs to group g = floor(i/(Cin/G)). Its kernel reaches only Cout/G output channels. Weights have shape [Cin,Cout/G,Kd,Kh,Kw] and are shared across every observation and spatial location.")
                        Text("For each axis: Lout = (Lin−1)·s − 2p + d·(K−1) + output_padding + 1").font(.system(size: 12, design: .monospaced)).textSelection(.enabled)
                        Text("All spatial triples use [depth,height,width]. Kernel[3,3,3], stride[2,2,2], padding[1,1,1] and output_padding[0,0,0] map 2×3×4 to 3×5×7. Change output padding to [0,0,1] for 3×5×8, then update the classifier width from 420 to 480. Output padding selects output geometry; it does not append zero-valued planes.")
                        Text("A width-3 kernel moving by stride 2 produces interior overlap counts of one or two along an axis. Across three axes, separable combinations can contribute 1, 2, 4 or 8 input positions per connected channel. Boundaries crop contributions. These counts describe geometry; actual responses also depend on inputs, learned weights and bias.")
                        Text("With matched weights, geometry and boundary conventions, the bias-free linear operator is the adjoint of convolution: ⟨Ax,y⟩ = ⟨x,Aᵀy⟩. It is not generally an inverse and cannot guarantee recovery of information lost by striding. Bias makes the operation affine; independently learned encoder and decoder weights are not automatically tied.")
                        Text("Groups=2 yields W[2,2,3,3,3]: 108 weights and four optional biases. Projection, transposed convolution and classifier total 1,098 parameters with bias. Try groups=1 to double the kernel weights. Change one spatial setting and inspect connector shapes; compatible checkpoints and refreshed previews are required after architecture changes.")
                        Text("This volume is constructed from ordinary measurements, not a scan, video or other volumetric source. Real volume data requires meaningful axes, spacing, scaling and preparation. This example teaches operator geometry without asserting domain suitability.")
                    }.font(.caption).foregroundStyle(MLTheme.muted).padding(.top, 6)
                }
            }
            Spacer()
            Button("Open volume transposed convolution example") { store.loadConvTranspose3dWorkflowExample() }.disabled(store.isBusy)
        }
    }
}
