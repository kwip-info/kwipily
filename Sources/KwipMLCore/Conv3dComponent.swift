import Foundation

/// Exact channels-first configuration is owned by the authored convolution bundle.
public enum Conv3dComponent {
    public static let definition = BundledConvolutionComponents.definition("conv3d")
    public static var metadata: BundledConvolutionComponents.Entry? { BundledConvolutionComponents.metadata("conv3d") }
}
