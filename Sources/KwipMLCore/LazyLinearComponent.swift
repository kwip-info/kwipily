import Foundation

/// Input width comes from the connected tensor; the bundle owns output and bias settings.
public enum LazyLinearComponent {
    public static let definition = BundledLayerComponents.definition("lazy_linear")
    public static var metadata: BundledLayerComponents.Entry? { BundledLayerComponents.entries["lazy_linear"] }
}
