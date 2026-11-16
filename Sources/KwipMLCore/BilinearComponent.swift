import Foundation

/// Production authoring comes from the Layers bundle. The measured lesson keeps
/// its separate supplied-weight and supplied-bias ports intentionally.
public enum BilinearComponent {
    public static let definition = BundledLayerComponents.definition("bilinear")
    public static var metadata: BundledLayerComponents.Entry? { BundledLayerComponents.entries["bilinear"] }
}
