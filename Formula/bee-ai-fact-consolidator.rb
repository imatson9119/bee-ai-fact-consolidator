class BeeAiFactConsolidator < Formula
  include Language::Python::Virtualenv

  desc "A tool to consolidate and deduplicate facts from Bee AI using clustering and a local LLM"
  homepage "https://github.com/imatson9119/bee-ai-fact-consolidator"
  url "https://github.com/imatson9119/bee-ai-fact-consolidator/archive/refs/tags/v0.2.0.tar.gz"
  sha256 "ee1266d975914b80c62d1237e968ad60a0f6de34ff5e4bf52eb016c674dc6a8d"
  license "MIT"

  depends_on "python@3.9"

  resource "requests" do
    url "https://files.pythonhosted.org/packages/9d/be/10918a2eac4ae9f02f6cfe6414b7a155ccd8f7f9d4380d62fd5b955065c3/requests-2.31.0.tar.gz"
    sha256 "942c5a758f98d790eaed1a29cb6eefc7ffb0d1cf7af05c3d2791656dbd6ad1e1"
  end

  resource "scikit-learn" do
    url "https://files.pythonhosted.org/packages/b0/dc/68ce32c94f2d607e378c5d714e3e943129f413e2fc1c834d6240c20c6d2d/scikit-learn-1.4.0.tar.gz"
    sha256 "8b0670d4224a3c2d596fd572fb4fa673b2a0ccfb07cc7f0524d8c7e4c430ddfa"
  end

  resource "python-dotenv" do
    url "https://files.pythonhosted.org/packages/31/06/1ef763af20d0572c032fa22882cfbfb005fba6e7300715a37840858c919e/python-dotenv-1.0.0.tar.gz"
    sha256 "a8df96034aae6d2d50a4ebe8216326c61c3eb64836776504fcca410e5937a3ba"
  end

  resource "openai" do
    url "https://files.pythonhosted.org/packages/5e/17/b1f1eaf6c7e87a6d93c5b4a6c9e7e1e6ad8d3fd5a379d0a0a9a3e4fe8d2d/openai-1.3.5.tar.gz"
    sha256 "3f2a5b4a30f7d10f6e8a962035e89f3e09a42b2e0a20229113f24d3ccbc23133"
  end

  resource "numpy" do
    url "https://files.pythonhosted.org/packages/a4/9b/027bec52c633f6556dba6b722d9a0befb40498b9ceddd29cbe67a45a127c/numpy-1.24.4.tar.gz"
    sha256 "80f5e3a4e498641401868df4208b74581206afbee7cf7b8329daae82676d9463"
  end

  resource "tqdm" do
    url "https://files.pythonhosted.org/packages/62/06/d5604a70d160f6a6ca5fd2ba25597c24abd5c5ca5f437263d177ac242308/tqdm-4.66.1.tar.gz"
    sha256 "d88e651f9db8d8551a62556d3cff9e3034274ca5d66e93197cf2490e2dcb69c7"
  end

  resource "click" do
    url "https://files.pythonhosted.org/packages/96/d3/f04c7bfcf5c1862a2a5b845c6b2b360488cf47af667dc468674b5c9d2591/click-8.1.7.tar.gz"
    sha256 "ca9853ad459e787e2192211578cc907e7594e294c7ccc834310722b41b9ca6de"
  end

  def install
    virtualenv_install_with_resources
  end

  test do
    system bin/"bee-fact-consolidator", "--help"
  end
end 