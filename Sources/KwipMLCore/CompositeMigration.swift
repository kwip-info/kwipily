import Foundation

public enum CompositeMigration {
  /// Pinned definition bodies retain their original representation and version; expansion normalizes a copy.
  public static func toCurrent(_ project: CompositeProjectDocument, registry: ComponentRegistry)
    throws -> CompositeProjectDocument
  {
    guard [1, 2].contains(project.schemaVersion) else {
      throw GraphConfigurationError("Unsupported composite envelope version.")
    }
    var copy = project
    copy.root = try toCurrentBody(project.root, registry: registry)
    copy.schemaVersion = 2
    return copy
  }
  /// Use for an editable draft of a definition; publishing still requires a new immutable revision.
  public static func toCurrentBody(_ body: CompositeGraph, registry: ComponentRegistry) throws
    -> CompositeGraph
  {
    guard [1, 2].contains(body.configurationVersion ?? 1) else {
      throw GraphConfigurationError("Unsupported composite configuration version.")
    }
    var copy = body
    for i in copy.nodes.indices {
      copy.nodes[i].config = try GraphMigration.effectiveConfiguration(
        of: body.nodes[i], schemaVersion: body.configurationVersion ?? 1, registry: registry)
    }
    copy.configurationVersion = 2
    return copy
  }
}
