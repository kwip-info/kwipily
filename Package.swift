// swift-tools-version: 6.1
import PackageDescription
import Foundation
let directUpdates = ProcessInfo.processInfo.environment["KWIPML_DIRECT_UPDATES"] == "1"
let stagingCommerce = ProcessInfo.processInfo.environment["KWIPML_COMMERCE_STAGING"] == "1"
let commerceSettings: [SwiftSetting] = stagingCommerce ? [.define("KWIPML_COMMERCE_STAGING")] : []
let products: [Product] = [
    .executable(name: "KwipMLMCPService", targets: ["KwipMLMCPService"]),
    .executable(name: "KwipMLMCPProbe", targets: ["KwipMLMCPProbe"]),
    .library(name: "KwipMLCore", targets: ["KwipMLCore"]),
    .executable(name: "KwipML", targets: ["KwipMLApp"]),
    .executable(name: "KwipMLNetworkService", targets: ["KwipMLNetworkService"]),
    .executable(name: "KwipMLDatabaseSourceService", targets: ["KwipMLDatabaseSourceService"]),
    .executable(name: "KwipMLDatabaseSourceProbe", targets: ["KwipMLDatabaseSourceProbe"]),
    .executable(name: "KwipMLIsolationProbe", targets: ["KwipMLIsolationProbe"])
]
let applicationDependencies: [Target.Dependency] = ["KwipMLMCP", "KwipMLCore", "KwipMLNetworking", "KwipMLDatabaseNetworking"] + (directUpdates ? [.target(name: "Sparkle")] : [])
let applicationSettings: [SwiftSetting] = commerceSettings + (directUpdates ? [.define("KWIPML_DIRECT_UPDATES")] : [])
var targets: [Target] = [
    .target(name: "KwipMLMCP", dependencies: ["KwipMLCore", .product(name: "MCP", package: "swift-sdk")]),
    .executableTarget(name: "KwipMLMCPService", dependencies: ["KwipMLMCP"]),
    .executableTarget(name: "KwipMLMCPProbe", dependencies: ["KwipMLMCP", "KwipMLCore"]),
    .testTarget(name: "KwipMLMCPTests", dependencies: ["KwipMLMCP"]),
    .target(name: "KwipMLCore", resources: [.process("Resources")], swiftSettings: commerceSettings),
    .target(name: "KwipMLNetworking", dependencies: ["KwipMLCore", "KwipMLDatabaseNetworking"]),
    .target(name: "KwipMLDatabaseNetworking", dependencies: ["KwipMLCore"]),
    .executableTarget(name: "KwipMLDatabaseSourceService", dependencies: ["KwipMLCore", "KwipMLDatabaseNetworking"]),
    .executableTarget(name: "KwipMLDatabaseSourceProbe", dependencies: ["KwipMLCore", "KwipMLDatabaseNetworking"]),
    .executableTarget(name: "KwipMLIsolationProbe", dependencies: ["KwipMLNetworking"]),
    .executableTarget(name: "KwipMLNetworkService", dependencies: ["KwipMLNetworking", "KwipMLCore"]),
    .executableTarget(name: "KwipMLApp", dependencies: applicationDependencies, resources: [.process("Resources")], swiftSettings: applicationSettings),
    .testTarget(name: "KwipMLNetworkingTests", dependencies: ["KwipMLNetworking", "KwipMLCore"]),
    .testTarget(name: "KwipMLCoreTests", dependencies: ["KwipMLCore"], swiftSettings: commerceSettings),
    .testTarget(name: "KwipMLAppTests", dependencies: ["KwipMLApp", "KwipMLCore"] + (directUpdates ? [.target(name: "Sparkle")] : []), swiftSettings: directUpdates ? [.define("KWIPML_DIRECT_UPDATES")] : [])
]
if directUpdates { targets.insert(.binaryTarget(name: "Sparkle", path: stagingCommerce ? ".local/sparkle-staging/Sparkle.xcframework" : ".local/sparkle/Sparkle.xcframework"), at: 0) }
let package = Package(name: "KwipML", platforms: [.macOS(.v14)], products: products, dependencies: [.package(url: "https://github.com/modelcontextprotocol/swift-sdk.git", exact: "0.12.1")], targets: targets, swiftLanguageModes: [.v5])
