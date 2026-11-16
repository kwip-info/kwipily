import Foundation

/// Optional exact discovery for explicit recipes. Legacy implicit Adam grants no recipe permission.
public struct OptimizerRuntimeCapabilities:Equatable {
    public static let operations=OptimizerConfiguration.componentIDs
    public private(set) var componentIDs:Set<String>=[]
    public init(descriptors:[String:Data]=[:]) {
        for id in Self.operations {
            if let data=descriptors[id],(try? Self.verify(data,componentID:id)) != nil {componentIDs.insert(id)}
        }
    }
    public func contains(_ id:String)->Bool {componentIDs.contains(id)}
    public static func verify(_ data:Data,componentID:String)throws {
        guard data.count<=8192,operations.contains(componentID),let entry=BundledOptimizerComponents.entries[componentID],entry.published,
              try JSONDecoder().decode(JSONValue.self,from:data) == entry.runtimeContract else {
            throw GraphConfigurationError("Optimizer support differs from its reviewed recipe, parameter ownership, controls or continuation policy.")
        }
    }
}
