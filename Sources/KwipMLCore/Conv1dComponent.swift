import Foundation

/// Exact channels-first configuration is owned by the authored convolution bundle.
public enum Conv1dComponent {
    public static let definition = BundledConvolutionComponents.definition("conv1d")
    public static var metadata: BundledConvolutionComponents.Entry? { BundledConvolutionComponents.metadata("conv1d") }
}
