import Foundation

/// Exact channels-first configuration is owned by the authored convolution bundle.
public enum Conv2dComponent {
    public static let definition = BundledConvolutionComponents.definition("conv2d")
    public static var metadata: BundledConvolutionComponents.Entry? { BundledConvolutionComponents.metadata("conv2d") }
}
