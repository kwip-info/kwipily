import Foundation

/// Exact lookup and integer boundary definitions are owned by the authored bundle.
public enum EmbeddingComponent {
    public static let indexInputDefinition = BundledEmbeddingComponents.definition("index_input")
    public static let definition = BundledEmbeddingComponents.definition("embedding")
    public static var metadata: BundledEmbeddingComponents.Entry? { BundledEmbeddingComponents.entries["embedding"] }
    public static var indexInputMetadata: BundledEmbeddingComponents.Entry? { BundledEmbeddingComponents.entries["index_input"] }
}
