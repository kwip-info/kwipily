import KwipMLCore
import SwiftUI

struct ExecutionContractView: View {
  let contract: BuiltInGraphExecutionContract

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text(
        "Current training task · \(contract.task.inputFeatures) input features · \(contract.task.outputFeatures) classes"
      )
      .font(.system(size: 12, weight: .medium))
      DisclosureGroup("Supported model contract") {
        VStack(alignment: .leading, spacing: 8) {
          Text(
            "Tensors use [batch, features] and \(contract.tensor.dtype) values (32-bit floating-point numbers). Batches must contain at least one observation. Dense tensors may be noncontiguous; row order is preserved."
          )
          Text(
            "Linear layers can include or omit bias. Each layer owns its parameters; changing bias ownership requires compatible weights."
          )
          Text(
            "Configured layer widths: 1–\(contract.limits.configuredWidth). Graph limit: \(contract.limits.nodes) blocks, \(contract.limits.edges) connections and \(contract.limits.parameters.formatted()) parameters."
          )
          Text(
            "Exact resume restores a matching CPU run at an epoch boundary. Fine-tuning reuses compatible weights with a fresh optimizer on CPU or an available Apple GPU."
          )
          Text(
            "Nonfinite source values, parameters and reported evaluation outputs are rejected. Exported Python source uses ordinary PyTorch behavior outside this app."
          )
        }.font(.system(size: 12)).foregroundStyle(MLTheme.muted).padding(.top, 6)
      }.font(.system(size: 12))
    }.frame(maxWidth: 620, alignment: .leading)
  }
}
