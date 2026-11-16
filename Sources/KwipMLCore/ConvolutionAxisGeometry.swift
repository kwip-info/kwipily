import Foundation

/// Per-axis padding and output arithmetic shared by bounded convolution contracts.
struct ConvolutionAxisGeometry {
    let before: Int
    let after: Int
    let output: Int
    init(length: Int, kernel: Int, stride: Int, dilation: Int, padding: Int, policy: String, mode: String, operation: String) throws {
        let effectiveKernel = dilation * (kernel - 1) + 1
        let before: Int, after: Int
        switch policy {
        case "same": before = (effectiveKernel - 1) / 2; after = effectiveKernel - 1 - before
        case "valid": before = 0; after = 0
        default: before = padding; after = padding
        }
        guard before <= 4096, after <= 4096 else { throw GraphConfigurationError("\(operation) resolved padding may not exceed 4096 on either side.") }
        if mode == "reflect", max(before, after) >= length {
            throw GraphConfigurationError("\(operation) reflection padding must be smaller than the input length on each side.")
        }
        if mode == "circular", max(before, after) > length {
            throw GraphConfigurationError("\(operation) circular padding may not exceed the input length on either side.")
        }
        let numerator = length + before + after - effectiveKernel
        // Reject before Swift truncates negative division toward zero.
        guard numerator >= 0 else { throw GraphConfigurationError("\(operation)'s effective kernel exceeds the padded input length. Reduce kernel/dilation or increase padding.") }
        let output = numerator / stride + 1
        guard (1...4096).contains(output) else { throw GraphConfigurationError("\(operation) output length must be from 1–4096.") }
        self.before = before; self.after = after; self.output = output
    }
}
