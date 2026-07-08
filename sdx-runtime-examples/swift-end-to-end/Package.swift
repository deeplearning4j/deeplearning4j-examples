// swift-tools-version: 5.9
import PackageDescription

// End-to-end SDX runtime example. The path dependency assumes a sibling
// `deeplearning4j` checkout next to this examples repository; when building
// against an unpacked SDK package, point it at <sdk>/wrappers/swift instead.
let package = Package(
    name: "SdxEndToEnd",
    platforms: [
        .macOS(.v13)
    ],
    dependencies: [
        .package(path: "../../../deeplearning4j/libnd4j/include/dsp/runtime/bindings/swift")
    ],
    targets: [
        .executableTarget(
            name: "SdxEndToEnd",
            dependencies: [
                .product(name: "SdxRuntime", package: "swift")
            ],
            path: "Sources/SdxEndToEnd"
        )
    ]
)
