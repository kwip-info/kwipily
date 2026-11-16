import Foundation

public struct MaxPool1dRuntimeCapability: Equatable {
    private init() {}
    public static func decode(_ data:Data) throws -> Self {
        try PoolingRuntimeCapabilities.verify(data, componentID: "max_pool1d")
        return Self()
    }
}
