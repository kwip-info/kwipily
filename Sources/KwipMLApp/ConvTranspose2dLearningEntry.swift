import SwiftUI

/// Spatial scatter geometry explored through a runnable prepared classifier.
struct ConvTranspose2dLearningEntry: View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store: WorkspaceStore
    var body: some View {
        HStack(alignment: .top, spacing: 24) {
            VStack(alignment: .leading, spacing: 8) {
                if !inLessonRail {
                Text("Transposed convolution on a constructed grid").font(.title3.weight(.semibold))
                Text("Project two measurements into 24 features, reshape to [N,2,3,4], and scatter grouped kernels into [N,4,6,7]. ReLU and flatten expose 168 responses to the classifier.").font(.system(size: 13)).foregroundStyle(MLTheme.muted)
                }
                DisclosureGroup("Understand spatial scatter, overlap and output geometry") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Y[n,g·(Cout/G)+o,h·sH−pH+u·dH,w·sW−pW+v·dW] += X[n,i,h,w] W[i,o,u,v]").font(.system(size: 12, design: .monospaced)).textSelection(.enabled)
                        Text("Start from zero, scatter each weighted input into valid output coordinates, and sum overlapping contributions. Then add one bias per output channel. Input channel i belongs to group g = floor(i/(Cin/G)); its kernel reaches only Cout/G output channels. Weights have shape [Cin,Cout/G,Kh,Kw] and are shared across positions and observations.")
                        Text("For each axis: Lout = (Lin−1)·s − 2p + d·(K−1) + output_padding + 1").font(.system(size: 12, design: .monospaced)).textSelection(.enabled)
                        Text("All pairs use [height,width]. Kernel[3,3], stride[2,2], padding[1,1] and output_padding[1,0] map 3×4 to 6×7. Change output padding to [0,0] for 5×7, then update the classifier input width from 168 to 140. Output padding selects among shapes compatible with a strided forward operation; it does not append zero-valued rows or columns.")
                        Text("For interior positions with a 3-wide kernel and stride 2, one axis alternates between one and two contributing input positions. Across two axes the counts can be 1, 2 or 4 per connected channel; boundaries crop contributions further. Inspect these overlap counts separately from learned values. Unequal counts do not imply a fixed output pattern because input values, weights and bias also matter.")
                        Text("With matched weights, geometry and boundary conventions, the bias-free linear operator is the adjoint of convolution: ⟨Ax,y⟩ = ⟨x,Aᵀy⟩. It is not generally an inverse. Bias makes the operation affine, and separately learned encoder and decoder weights are not automatically tied.")
                        Text("Groups=2 gives W[2,2,3,3]: 36 weights and four optional biases. Projection, transposed convolution and classifier total 450 parameters with bias. Try groups=1 to double kernel weights, or edit one spatial setting and inspect which connector dimension changes. Checkpoints and cached previews must remain compatible with the architecture.")
                        Text("These grids are learned constructions from ordinary measurements, not source images. Real image work needs meaningful channel and pixel ordering, scaling and preparation. This example teaches spatial geometry and learned operators without claiming image-domain suitability.")
                    }.font(.caption).foregroundStyle(MLTheme.muted).padding(.top, 6)
                }
            }
            Spacer()
            Button("Open spatial transposed convolution example") { store.loadConvTranspose2dWorkflowExample() }.disabled(store.isBusy)
        }
    }
}
