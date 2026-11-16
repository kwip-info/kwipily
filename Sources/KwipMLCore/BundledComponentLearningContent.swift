/// Shared display contract for authored component learning across bundle families.
/// Concrete generated records retain their own manifest/runtime identities.
public protocol BundledComponentLearningContent {
    var contentVersion: Int { get }
    var updatedAt: String { get }
    var exampleID: String { get }
    var sourceResource: String { get }
    var title: String { get }
    var subtitle: String { get }
    var disclosureTitle: String { get }
    var formula: String { get }
    var paragraphs: [String] { get }
    var buttonTitle: String { get }
}
