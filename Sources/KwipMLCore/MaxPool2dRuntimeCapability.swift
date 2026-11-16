import Foundation

public struct MaxPool2dRuntimeCapability: Equatable {
    private init() {}
    public static func decode(_ data:Data) throws -> Self {
        try PoolingRuntimeCapabilities.verify(data, componentID: "max_pool2d")
        return Self()
    }
}
