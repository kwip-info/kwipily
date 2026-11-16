import SwiftUI

/// A compact exploration entry backed by the runnable prepared example.
struct ConvTranspose1dLearningEntry: View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store: WorkspaceStore
    var body: some View {
        HStack(alignment:.top,spacing:24) {
            VStack(alignment:.leading,spacing:8) {
                if !inLessonRail {
                Text("Transposed convolution on a constructed signal").font(.title3.weight(.semibold))
                Text("Project two measurements into 14 features, reshape to [N,2,7], and expand length to 14 with grouped transposed kernels. ReLU and flatten expose 56 responses to the classifier.").font(.system(size:13)).foregroundStyle(MLTheme.muted)
                }
                DisclosureGroup("Understand scatter-add, adjoints and output geometry") {
                    VStack(alignment:.leading,spacing:8) {
                        Text("Y[n,g·(Cout/G)+o,t·s−p+k·d] += X[n,i,t] W[i,o,k]").font(.system(size:12,design:.monospaced)).textSelection(.enabled)
                        Text("Start the output at zero. Each input channel i belongs to group g = floor(i/(Cin/G)); scatter its weighted kernel contributions into valid output positions, sum overlaps, then add one bias per output channel. Weights are shared across observations and positions. Unlike Conv1d, their layout is [Cin,Cout/G,K].")
                        Text("Lout = (Lin−1)·s − 2p + d·(K−1) + output_padding + 1").font(.system(size:12,design:.monospaced)).textSelection(.enabled)
                        Text("Here (7−1)·2 − 2·1 + 1·(3−1) + 1 + 1 = 14. With stride greater than one, several input lengths of a forward convolution can map to the same output length. Output padding selects the transposed output extent; it does not append a row of zero values. Set it to zero to get length 13 and update the classifier input width from 56 to 52.")
                        Text("With matched weights, geometry and boundary conventions, the bias-free linear operator is the matrix transpose (adjoint) of convolution: ⟨Ax,y⟩ = ⟨x,Aᵀy⟩. It is not generally an inverse and cannot restore information lost by striding. Adding learned bias makes the operation affine. Independently learned decoder weights are not automatically tied to an encoder.")
                        Text("Groups=2 gives W[2,2,3]: 12 weights and four optional biases. Including projection and classifier, this example has 172 parameters. Try groups=1 to double the kernel weights, or change stride and inspect connector shapes. Architecture changes require compatible checkpoints and refreshed caches.")
                        Text("The signal is a learned construction from ordinary measurements, not recorded audio or a time series. Real sequence work needs meaningful ordering, scaling and preparation. This example teaches the operation and geometry; it makes no claim about suitability for a particular data domain.")
                    }.font(.caption).foregroundStyle(MLTheme.muted).padding(.top,6)
                }
            }
            Spacer()
            Button("Open transposed convolution example") {store.loadConvTranspose1dWorkflowExample()}.disabled(store.isBusy)
        }
    }
}
