// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "DiffusionKit",
    platforms: [
        .macOS("13.3"),
        .iOS(.v16),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "DiffusionKit",
            targets: ["DiffusionKit"]
        ),
        .library(
            name: "DiffusionKitMLX",
            targets: ["DiffusionKitMLX"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers.git", exact: "0.1.8"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", exact: "1.3.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", branch: "main"),
    ],
    targets: [
        .target(
            name: "DiffusionKit",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers")
            ],
            path: "swift/Sources/DiffusionKit"
        ),
        .target(
            name: "DiffusionKitMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "swift/Sources/DiffusionKitMLX"
        ),
        .testTarget(
            name: "DiffusionKitTests",
            dependencies: [
                "DiffusionKit",
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "swift/Tests/DiffusionKitTests"
        ),
        .testTarget(
            name: "DiffusionKitMLXTests",
            dependencies: [
                "DiffusionKitMLX",
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "swift/Tests/DiffusionKitMLXTests"
        ),
    ]
)
